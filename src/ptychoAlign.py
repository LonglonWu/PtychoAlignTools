"""
Probe Alignment
2 images alignment

Requirements:
matplotlib
pyqtgraph
tifffile
numpy

V01 Notes:
1) The path for the image files is hard coded. Need to change to the path where you will put the images.
See module (function) loadProbeImage.

2) Ignore the "delPosition" and the "reconstruction" functions. I'm working on this functions yet.

3) d = np.zeros((11,11,2),dtype=int) - Data Structure to store all the relative positions

V02 Notes:
1)Load and Save Relative Positions using .npy files


V03 Notes (22/01/2018):
1) Load mask
2) GK operation with density maps

V04 Notes (01/02/2018):
1)Preview global picture

V05 Notes (02/02/2018):
1) displayImages() renamed -> refreshView()
2) button title "Preview" renamed -> "Preview Global Image"
3) Update currentCell of tablePositions when clicking on probes to be aligned
4) 
"""

import os
import tempfile
import time
os.environ['PYQTGRAPH_QT_LIB']='PyQt5' #Garantee pyqtgraph uses PyQt5 if avalilable
import matplotlib.image as mpimg

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *
from pyqtgraph import widgets

import tifffile as tf
import numpy as np


class PtychoAlign(object):

    def __init__(self):
        
        self.app = None

        self.winMain = None # Main window to contain the dock area
        self.winPreview = None # Window show on displaying Preview global picture

        self.fpath = '' # variables used to open a file with positions
        self.fname = ''
        self.fformat = ''
        
        self.probes_paths = None # list containing the paths of the probe scans
        
        self.maskpath = '' # stores the mask file path, name and format
        self.maskname = ''
        self.maskformat = ''

        self.hor_ver_header = [] # horizontal and vertical header for Table
        
        self.plt_align = None # Plot area for the alignment
        self.plt_probe = None # Plot area for the probe image

        self.mask = np.ones((1,1)) # Mask image data numpy array

        # user input
        self.col = None # number of columns of the scan map
        self.row = None # number of rows of the scan map
        
        self.img_anchored = None # Image Data numpy array
        self.img_movable = None

        self.img_align = None # Image Item
        self.img_probe = None
        
        self.canvas_anchored = None # Numpy arrays
        self.canvas_movable = None
        self.canvas_probe = None

        self.anchored = 0 # initialinzg the anchored and movable probes
        self.movable = 1

        self.ref = None  # reference to build the whole canvas

        self.data = [] # list containing the loaded image (numpy arrays)
        self.map_pos = []

        # Initialize relative positions with zeros (no alignments)
##        self.rel_pos = np.zeros((12, 12, 2)) # Data structure shape
        self.rel_pos = np.zeros((1,1))

        self.roi_probe_anc = None # red roi 
        self.roi_probe_mov = None # green roi         
        

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # New application instance
        self.app = QtGui.QApplication([])
        self.app.setWindowIcon(QtGui.QIcon('./pictures/icon.png'))
        PyQt_version = str(self.app)
        print type(PyQt_version), PyQt_version

        # Window for preview
        self.winPreview = QtGui.QMainWindow()
        self.winPreview.resize(800,800)
        self.imvPreview = pg.ImageView()
        self.winPreview.setCentralWidget(self.imvPreview)
        
        # Main Window
        self.winMain = QtGui.QMainWindow()
        area = DockArea()
        area.setParent(self.winMain)
        self.winMain.setCentralWidget(area)
        self.winMain.resize(1200,1200)
        self.winMain.setWindowTitle('PtychoAlign - ')

        # QDialog for user input
        self.dialog = QtGui.QDialog(self.winMain)
        self.form = QtGui.QFormLayout(self.dialog)
        self.form.addRow(QtGui.QLabel("Please inform number of rows and columns of your scan map."))
        self.colLineEdit = QtGui.QLineEdit("0")
        self.rowLineEdit = QtGui.QLineEdit("0")
        self.form.addRow(QtGui.QLabel("Col"), self.colLineEdit)
        self.form.addRow(QtGui.QLabel("Row"), self.rowLineEdit)
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok |
                                                QtGui.QDialogButtonBox.Cancel,
                                                QtCore.Qt.Horizontal,
                                                self.dialog)
        button_ok = self.buttonBox.button(QtGui.QDialogButtonBox.Ok)
        button_cancel = self.buttonBox.button(QtGui.QDialogButtonBox.Cancel)
        button_ok.clicked.connect(self.dialogOk)
        button_cancel.clicked.connect(self.dialogCancel)
        self.form.addRow(self.buttonBox)
        
        d1 = Dock("Tools", size=(400,600))
        d2 = Dock("Single Reconstructed Probes", size=(800,600))
        d3 = Dock("Positions", size=(600,600)) 
        d4 = Dock("Pairwise Alignment View", size=(600,600))

        area.addDock(d1, 'left')
        area.addDock(d2, 'right')
        area.addDock(d3, 'bottom', d1)
        area.addDock(d4, 'bottom', d2)

        
        """Window for Alignment"""
##        winAlign = pg.GraphicsWindow()
        self.winAlign = pg.GraphicsLayoutWidget(parent=self.winMain)        
        self.plt_align = self.winAlign.addPlot()
        self.img_align = pg.ImageItem()
        self.plt_align.addItem(self.img_align)
        """Contrast/color control"""
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_align)
        self.winAlign.addItem(self.hist)

        """Window to display the whole Probe"""
        winProbe = pg.GraphicsLayoutWidget(parent=self.winMain)
        self.plt_probe = winProbe.addPlot()
        self.img_probe = pg.ImageItem()
        self.plt_probe.addItem(self.img_probe)

        """Table relative Positions [anchored movable (x,y)]"""
        self.tablePositions = pg.TableWidget(editable=False, sortable=False)       
        
        winTools = QtGui.QWidget(parent=self.winMain)
##        winTools.setFocusProxy(self.winAlign)
##        winTools = QtGui.QMainWindow()
        

        """Tools"""
            
        # Buttons
        group = QtGui.QGroupBox(" ")

        button_load_probes = QtGui.QPushButton("Load Probes")
        button_load_probes.clicked.connect(self.openProbes)
        
        button_load_align = QtGui.QPushButton("Load Alignment")
        button_load_align.clicked.connect(self.openAlign)

        button_save_align = QtGui.QPushButton("Save Alignment")
        button_save_align.clicked.connect(self.saveAlign)

        button_refresh = QtGui.QPushButton("&Refresh View")
        button_refresh.clicked.connect(self.refreshView)

        button_preview = QtGui.QPushButton("&Preview Global Image")
        button_preview.clicked.connect(self.previewRec)
        
        self.alignment_direction = QtGui.QButtonGroup(parent=winTools)
##        self.alignement_direction.setExclusive(True)
        check_horizontal = QtGui.QCheckBox("Horizontal Alignment")
        check_horizontal.setChecked(True)
        check_vertical = QtGui.QCheckBox("Vertical Alignment")
        self.alignment_direction.addButton(check_horizontal)
        self.alignment_direction.addButton(check_vertical)
        
        self.linedit_anchored = QtGui.QLineEdit()
        self.linedit_anchored.setAlignment(QtCore.Qt.AlignHCenter)
        self.linedit_movable = QtGui.QLineEdit()
        self.linedit_movable.setAlignment(QtCore.Qt.AlignHCenter)        

        button_canvas_anchored = QtGui.QPushButton("&Anchored Image")
        button_canvas_anchored.setStyleSheet("background-color: red; color: black")
        button_canvas_anchored.clicked.connect(self.createCanvasAnchored)
        button_canvas_movable = QtGui.QPushButton("&Movable Image")
        button_canvas_movable.setStyleSheet("background-color: green; color: black")
        button_canvas_movable.clicked.connect(self.createCanvasMovable)        

        label_operation = QtGui.QPushButton("Operation")
        label_operation.setFlat(True)
        label_operation.setFocusPolicy(QtCore.Qt.NoFocus)
        label_operation.setFocusPolicy(QtCore.Qt.NoFocus)
        self.combo_operation = QtGui.QComboBox()        
        self.combo_operation.addItems(["Multiplication", "Division", "Addition", "Subtraction", "Mask"])

        label_weight_1 = QtGui.QPushButton("W1")
        label_weight_1.setFlat(True)
        label_weight_1.setFocusPolicy(QtCore.Qt.NoFocus)        
        self.spin_weight_1 = QtGui.QDoubleSpinBox()
        self.spin_weight_1.setValue(1)
        self.spin_weight_1.setDecimals(1)
        self.spin_weight_1.setRange(0.1, 1.0)
        self.spin_weight_1.setSingleStep(0.1)
        self.spin_weight_1.setAlignment(QtCore.Qt.AlignHCenter)

        label_short_step = QtGui.QPushButton("Short Step \n[Up, Left, Down, Right] keys")
        label_short_step.setFlat(True)
        label_short_step.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin_short_step = QtGui.QSpinBox()
        self.spin_short_step.setValue(1)
        self.spin_short_step.setRange(1,100)
        self.spin_short_step.setAlignment(QtCore.Qt.AlignHCenter)

        label_large_step = QtGui.QPushButton("Large Step \n[W,A,S,D] keys")
        label_large_step.setFlat(True)
        label_large_step.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin_large_step = QtGui.QSpinBox()
        self.spin_large_step.setValue(5)
        self.spin_large_step.setRange(5,100)
        self.spin_large_step.setSingleStep(5)
        self.spin_large_step.setAlignment(QtCore.Qt.AlignHCenter)

        label_position_horizontal = QtGui.QPushButton("Position X")
        label_position_horizontal.setFlat(True)
        label_position_horizontal.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin_position_horizontal = QtGui.QSpinBox()
        self.spin_position_horizontal.setKeyboardTracking(False)
        self.spin_position_horizontal.setRange(-10000, 10000)
        self.spin_position_horizontal.setAlignment(QtCore.Qt.AlignHCenter)
        
        label_position_vertical = QtGui.QPushButton("Position Y")
        label_position_vertical.setFlat(True)
        label_position_vertical.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spin_position_vertical = QtGui.QSpinBox()
        self.spin_position_vertical.setKeyboardTracking(False)
        self.spin_position_vertical.setRange(-10000, 10000)
        self.spin_position_vertical.setAlignment(QtCore.Qt.AlignHCenter)

        button_load_mask = QtGui.QPushButton("Load Mask [Mask operation]")
        button_load_mask.clicked.connect(self.openMask)
        self.label_mask = QtGui.QPushButton("Mask name")
        self.label_mask.setFlat(True)
        self.label_mask.setFocusPolicy(QtCore.Qt.NoFocus)
        

        """Layouts"""

        """Horizontal Layouts"""
        # Anchored and Movable
        layout_anc_mov = QtGui.QHBoxLayout()
        layout_anc_mov.addWidget(self.linedit_anchored)
        layout_anc_mov.addWidget(self.linedit_movable)

        # Create Canvas
        layout_canvas_tb = QtGui.QHBoxLayout() # tb - top bottom
        layout_canvas_tb.addWidget(button_canvas_anchored)
        layout_canvas_tb.addWidget(button_canvas_movable)
        layout_canvas_lr = QtGui.QHBoxLayout() # lr - left right

        # Operation
        layout_operation = QtGui.QHBoxLayout()
        layout_operation.addWidget(label_operation)
        layout_operation.addWidget(self.combo_operation)
        layout_operation.addWidget(label_weight_1)
        layout_operation.addWidget(self.spin_weight_1)

        # Steps
        layout_steps = QtGui.QHBoxLayout()
        layout_steps.addWidget(label_short_step)
        layout_steps.addWidget(self.spin_short_step)
        layout_steps.addWidget(label_large_step)
        layout_steps.addWidget(self.spin_large_step)

        # Position
        layout_position = QtGui.QHBoxLayout()
        layout_position.addWidget(label_position_horizontal)
        layout_position.addWidget(self.spin_position_horizontal)
        layout_position.addWidget(label_position_vertical)
        layout_position.addWidget(self.spin_position_vertical)
        

        """Vertical Layouts"""
        layout = QtGui.QVBoxLayout()
        layout.addWidget(button_load_probes)
        layout.addWidget(button_load_align)
        layout.addWidget(button_save_align)
        layout.addWidget(button_refresh)
        layout.addWidget(button_preview)
        layout.addLayout(layout_canvas_tb)
        layout.addLayout(layout_canvas_lr)
        layout.addLayout(layout_anc_mov)        
        layout.addLayout(layout_operation)
        layout.addLayout(layout_steps)
        layout.addLayout(layout_position)
        layout.addWidget(button_load_mask)
        layout.addWidget(self.label_mask)
        group.setLayout(layout)

        toolsLayout = QtGui.QVBoxLayout()
        toolsLayout.addWidget(group)
        winTools.setLayout(toolsLayout)

        """Add Widgets to Dock Areas"""
        d1.addWidget(winTools)
        d2.addWidget(winProbe)
        d3.addWidget(self.tablePositions)
        d4.addWidget(self.winAlign)

        """Set up Table of Positions"""
        self.tablePositions.setData(self.rel_pos.tolist())
        self.tablePositions.setHorizontalHeaderLabels(self.hor_ver_header)
        self.tablePositions.setVerticalHeaderLabels(self.hor_ver_header)
        
        """Set up signals, key events, mouse events"""
        self.winAlign.keyPressEvent = self.keyPressEventAlignment
        winProbe.keyPressEvent = self.keyPressEventProbe
        group.keyPressEvent = self.keyPressEventButtons
        d3.keyPressEvent = self.keyPressEventButtons

        self.img_probe.mousePressEvent = self.mouseClickEventProbe
        self.tablePositions.cellClicked.connect(self.cellClicked)

        # Calls alignImage() every time the values in the spin boxes changes
        self.spin_position_horizontal.valueChanged['int'].connect(self.alignImage)
        self.spin_position_vertical.valueChanged['int'].connect(self.alignImage)

        self.combo_operation.currentIndexChanged.connect(self.alignImage)

    # key actions for aligning the images 
    def keyPressEventAlignment(self, event):      
        if type(event) == QtGui.QKeyEvent:              
            if event.key() == QtCore.Qt.Key_Left:
                self.spin_position_horizontal.setValue(self.spin_position_horizontal.value() - self.spin_short_step.value())                
            elif event.key() == QtCore.Qt.Key_Right:
                self.spin_position_horizontal.setValue(self.spin_position_horizontal.value() + self.spin_short_step.value())                
            elif event.key() == QtCore.Qt.Key_Up:                
                self.spin_position_vertical.setValue(self.spin_position_vertical.value() + self.spin_short_step.value())
            elif event.key() == QtCore.Qt.Key_Down:
                self.spin_position_vertical.setValue(self.spin_position_vertical.value() - self.spin_short_step.value())
            elif event.key() == QtCore.Qt.Key_A:
                self.spin_position_horizontal.setValue(self.spin_position_horizontal.value() - self.spin_large_step.value())
            elif event.key() == QtCore.Qt.Key_D:
                self.spin_position_horizontal.setValue( self.spin_position_horizontal.value() + self.spin_large_step.value())
            elif event.key() == QtCore.Qt.Key_W:
                self.spin_position_vertical.setValue(self.spin_position_vertical.value() + self.spin_large_step.value())
            elif event.key() == QtCore.Qt.Key_S:
                self.spin_position_vertical.setValue(self.spin_position_vertical.value() - self.spin_large_step.value())

                
    def keyPressEventProbe(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_S:
                print "Swap probes"
                self.swapProbes()
            elif event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
                self.refreshView()
                self.winAlign.setFocus()

    def keyPressEventButtons(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
                self.winAlign.setFocus()                

    def mouseClickEventProbe(self, event):
##        print help(event)
        print event.button()
        print event.buttons()
        print event.flags()
        print help(event.modifiers())
        mouse_x = round(event.pos().x())
        mouse_y = round(event.pos().y())

##        print mouse_x, mouse_y

        mult_x = int(mouse_x / self.data[0].shape[1])
        mult_y = int(mouse_y / self.data[0].shape[0])
        
##        "Set anchored image"
        if event.button() == 1:
##            print "Left Click"
            self.roi_probe_mov.setPos(mult_x * self.data[0].shape[1], mult_y * self.data[0].shape[0])            
            self.movable = self.map_pos.index((mult_x, mult_y))
            self.linedit_movable.setText(str(self.movable))
            self.text_mov.setText(str(self.movable), color='g')
            self.tablePositions.setCurrentCell(self.movable, self.anchored)
            self.createCanvasMovable()           

##        "Set movable image"
        elif event.button() == 2:
##            print "Right Click"
            self.roi_probe_anc.setPos(mult_x * self.data[0].shape[1], mult_y * self.data[0].shape[0])
            self.anchored = self.map_pos.index((mult_x, mult_y))
            self.linedit_anchored.setText(str(self.anchored))
            self.text_anc.setText(str(self.anchored), color='r')
            self.tablePositions.setCurrentCell(self.movable, self.anchored)
            self.createCanvasAnchored()
        
        self.spin_position_horizontal.setValue(self.rel_pos[self.movable, self.anchored, 0])
        self.spin_position_vertical.setValue(self.rel_pos[self.movable, self.anchored, 1])
##        self.refreshView()
        self.alignImage
        self.winAlign.setFocus()

    def cellClicked(self, row, col):
##        print row, col
        if self.map_pos == []:
            pass
        else:
            self.movable = row
            self.anchored = col
            movable = self.map_pos[row]
            anchored = self.map_pos[col]
    ##        print movable, anchored
            self.text_mov.setText(str(row), color='g')
            self.text_anc.setText(str(col), color='r')
            self.linedit_movable.setText(str(self.movable))
            self.linedit_anchored.setText(str(self.anchored))
            self.roi_probe_mov.setPos(movable[0] * self.data[0].shape[1], movable[1] * self.data[0].shape[0])
            self.roi_probe_anc.setPos(anchored[0]* self.data[0].shape[1], anchored[1] * self.data[0].shape[0])
            self.createCanvasMovable()
            self.createCanvasAnchored()
            x = self.rel_pos[self.movable, self.anchored, 0]
            y = self.rel_pos[self.movable, self.anchored, 1]
            self.spin_position_horizontal.setValue(x)
            self.spin_position_vertical.setValue(y)
##            self.refreshView()
            self.alignImage()
            self.tablePositions.resizeColumnsToContents()
##        self.winAlign.setFocus()

    def dialogOk(self):
        # try | except block to garantee only integers as input
        try:
            eval(str(self.colLineEdit.text()))
            eval(str(self.rowLineEdit.text()))
            if type(eval(str(self.colLineEdit.text()))) == int and type(eval(str(self.rowLineEdit.text()))) == int:                
                self.col = eval(str(self.colLineEdit.text()))
                self.row = eval(str(self.rowLineEdit.text()))
                self.dialog.accept()
            else:                
                reply = QtGui.QMessageBox.information(self.dialog, "Warning",
"""Please type only integers numbers.
""")            
                self.dialog.open()                
        except NameError:            
            reply = QtGui.QMessageBox.information(self.dialog, "Warning",
"""Please type only integers numbers.
""")            
            self.dialog.open()

    def dialogCancel(self):        
        self.dialog.reject()

    def openProbes(self):
        # ask for user input the shape of the probe scans
        self.dialog.exec_()
        # Opens a File Dialog
        if self.probes_paths != None:
            self.probes_paths = None        
        if self.dialog.result():
            self.probes_paths = QtGui.QFileDialog.getOpenFileNames(self.winMain, 'Load Probes',  '.',
                                                       "Images (*.tiff *.tif);;All (*.*)")
        else:
            pass
    ##            print self.probes_paths[1]         
        if self.probes_paths[0] == []:
            pass
        else:
            self.loadProbes()
##            print self.probes_paths[0][0][ :self.probes_paths[0][0].rindex('/') ]
            self.winMain.setWindowTitle('PtychoAlign - %s'
                                        % self.probes_paths[0][0][ :self.probes_paths[0][0].rindex('/') ])

    def loadProbes(self):
        self.data = []
        for i in range(len(self.probes_paths[0])):
            try:
                self.data.append(np.flipud(tf.imread(str(self.probes_paths[0][i]))))
            except ValueError:
                reply = QtGui.QMessageBox.information(self.winMain, "Warning",
"""The file loaded must be of tiff or tif format in order to load properly.
""")
##        self.col = 2
##        self.row = 2
##        self.canvas_probe = np.ones((self.data[0].shape[0]*4, self.data[0].shape[1]*3), dtype=np.float32)
        self.canvas_probe = np.ones((self.data[0].shape[0]*self.row, self.data[0].shape[1]*self.col), dtype=np.float32)
        print "data len", len(self.data)
        print "canvas_probe shape", self.canvas_probe.shape
        
        i = 0
        j = 0
        k = 0
        for d in range(len(self.data)):
            print i,j,k
            self.map_pos.append((j,i))
            if i == (self.row-1):
                self.canvas_probe[self.canvas_probe.shape[0]-(i+1)*self.data[0].shape[0]:
                                  self.canvas_probe.shape[0]-i*self.data[0].shape[0],
                                  self.canvas_probe.shape[1]-(j+1)*self.data[0].shape[1]:
                                  self.canvas_probe.shape[1]-j*self.data[0].shape[1]] = self.data[k]
                i = 0
                j += 1
                k += 1
                continue
            else:
                self.canvas_probe[self.canvas_probe.shape[0]-(i+1)*self.data[0].shape[0]:
                                  self.canvas_probe.shape[0]-i*self.data[0].shape[0],
                                  self.canvas_probe.shape[1]-(j+1)*self.data[0].shape[1]:
                                  self.canvas_probe.shape[1]-j*self.data[0].shape[1]] = self.data[k]
                i += 1
                k += 1

        self.map_pos = self.map_pos[::-1]
##        print self.map_pos
        self.img_probe.setImage(self.canvas_probe)

        self.mask = np.ones((self.data[0].shape[0], self.data[0].shape[1]))

        self.ref = 512
        self.ref_rec = 320
        # To avoid creating ROI objects twice 
        if self.roi_probe_anc == None:
            pen = pg.mkPen(color='r', width=2)
##            self.roi_probe_anc = pg.ROI([4096,6144], [self.data[0].shape[1], self.data[0].shape[0]], pen=pen, movable=False)
            self.roi_probe_anc = pg.ROI([self.data[0].shape[0]*(self.col-1),self.data[0].shape[1]*(self.row-1)],
                                        [self.data[0].shape[1], self.data[0].shape[0]], pen=pen, movable=False)        
            self.roi_probe_anc.setParentItem(self.img_probe)
            self.text_anc = pg.TextItem()
            self.text_anc.setParentItem(self.roi_probe_anc)
            self.text_anc.setText('0', color='r')
            
            pen = pg.mkPen(color='g', width=2)
##            self.roi_probe_mov = pg.ROI([4096,4096], [self.data[0].shape[1], self.data[0].shape[0]], pen=pen, movable=False)
            self.roi_probe_mov = pg.ROI([self.data[0].shape[0]*(self.col-1),self.data[0].shape[1]*(self.row-2)],
                                        [self.data[0].shape[1], self.data[0].shape[0]], pen=pen, movable=False)
            self.roi_probe_mov.setParentItem(self.img_probe)
            self.text_mov = pg.TextItem()
            self.text_mov.setParentItem(self.roi_probe_mov)
            self.text_mov.setText('1', color='g')
        else:
            self.roi_probe_anc.setSize([self.data[0].shape[1], self.data[0].shape[0]])
            self.roi_probe_anc.setPos([self.data[0].shape[0]*(self.col-1),self.data[0].shape[1]*(self.row-1)])
            self.roi_probe_mov.setSize([self.data[0].shape[1], self.data[0].shape[0]])
            self.roi_probe_mov.setPos([self.data[0].shape[0]*(self.col-1),self.data[0].shape[1]*(self.row-2)])
            self.text_anc.setText('0', color='r')
            self.text_mov.setText('1', color='g')

        
        self.rel_pos = np.zeros((self.col*self.row, self.col*self.row, 2))
        # setting table data
        self.tablePositions.setData(self.rel_pos.tolist())
        for i in range(self.col*self.row):
            self.hor_ver_header.append(str(i))            
        self.tablePositions.setHorizontalHeaderLabels(self.hor_ver_header)
        self.tablePositions.setVerticalHeaderLabels(self.hor_ver_header)        
        self.tablePositions.setCurrentCell(self.movable, self.anchored)

        self.createCanvasMovable()
        self.createCanvasAnchored()
        self.refreshView()        

##        self.rel_pos = np.zeros((len(self.data),len(self.data),2),dtype=int)
        
        self.plt_probe.autoRange()        

    def openAlign(self):
        print "Load Align"
        # Opens a File Dialog
        if self.fpath != '':
            self.fpath = ''
            self.fname = ''
            self.fformat = ''
        print os.curdir
        self.fpath = QtGui.QFileDialog.getOpenFileName(self.winMain, 'Load Alignment',  '.',
                                                       "(*.npy);;All (*.*)")        
        if self.fpath[0] != '':
            self.fpath = self.fpath[0]
            self.fname = self.fpath[self.fpath.rindex('/')+1::]
            self.fformat = self.fpath[self.fpath.rindex('.')+1::]
            self.loadAlign()

    def loadAlign(self):
        if self.fformat == 'npy':
            self.rel_pos = np.load(self.fpath)
            self.tablePositions.setData(self.rel_pos.tolist())
            self.tablePositions.setHorizontalHeaderLabels(self.hor_ver_header)
            self.tablePositions.setVerticalHeaderLabels(self.hor_ver_header)
            self.tablePositions.resizeColumnsToContents()
##            print self.rel_pos
        else:
            print "File format not supported!"
            reply = QtGui.QMessageBox.information(self.winMain, "Warning",
"""The file loaded must be of npy format in order to load properly.
""")

    def saveAlign(self):
        print "Save Align"
        save = QtGui.QFileDialog.getSaveFileName(self.winMain, 'Save Alignment', '.',
                                                     "(*.npy)")
        if save[0] != '':
            savepath = str(save[0])
            saveformat = str(save[1])
            saveformat = saveformat[saveformat.rindex('.'):-1]
            print savepath, saveformat
            np.save(savepath + saveformat, self.rel_pos)
        else:
            print "Not a valid file name!"
            reply = QtGui.QMessageBox.information(self.winMain, "Warning",
"""Please type a valid file name to save it properly.
""")

    def openMask(self):
        print "Load Mask"
        # Opens a File Dialog
        if self.maskpath != '':
            self.maskpath = ''
            self.maskname = ''
            self.maskformat = ''

        self.maskpath = QtGui.QFileDialog.getOpenFileName(self.winMain, 'Load Mask',  '.',
                                                       "(*.png);;(*.tif);;(*.tiff);;All (*.*)")        
        if self.maskpath[0] != '':            
            self.maskpath = self.maskpath[0]
            self.maskname = self.maskpath[self.maskpath.rindex('/')+1::]
            self.maskformat = self.maskpath[self.maskpath.rindex('.')+1::]
            self.label_mask.setText('%s' % self.maskname)
            self.loadMask()
            

    def loadMask(self):
        print self.maskpath, self.maskname, self.maskformat
        if self.maskformat == 'png':
            self.mask = mpimg.imread(self.maskpath)
        elif self.maskformat in ['tif', 'tiff']:
            self.mask = tf.imread(self.maskpath)
        else:
            print "Invalid format!"
            reply = QtGui.QMessageBox.information(self.winMain, "Warning",
"""The file loaded must be of png, tif or tiff format in order to load properly.
""")

        self.mask = self.mask.astype(np.float32)
        self.mask -= self.mask.min()
        self.mask /= self.mask.max()
        self.alignImage()

    def swapProbes(self):
        print "ROI's: mov, anc", self.movable, self.anchored
        print "canvas probe shape", self.canvas_probe.shape
        print "ROI's positions", self.roi_probe_mov.pos(), self.roi_probe_anc.pos()
        print "ROI's sizes", self.roi_probe_mov.size(), self.roi_probe_anc.size()

        pos_mov_x, pos_mov_y = int(self.roi_probe_mov.pos()[0]), int(self.roi_probe_mov.pos()[1])
        pos_anc_x, pos_anc_y = int(self.roi_probe_anc.pos()[0]), int(self.roi_probe_anc.pos()[1])       

        size_x = int(self.roi_probe_mov.size()[0])
        size_y = int(self.roi_probe_mov.size()[1])        
        
        swap_aux_anc = self.roi_probe_anc.getArrayRegion(self.canvas_probe, self.img_probe)
        swap_aux_mov = self.roi_probe_mov.getArrayRegion(self.canvas_probe, self.img_probe)
        
        self.canvas_probe[pos_mov_y:pos_mov_y + size_y,
                          pos_mov_x:pos_mov_x + size_x] = swap_aux_anc
        self.canvas_probe[pos_anc_y:pos_anc_y + size_y,
                          pos_anc_x:pos_anc_x + size_x] = swap_aux_mov
        self.img_probe.setImage(self.canvas_probe)
        
                          
    
    def createCanvasAnchored(self):
        self.canvas_anchored = np.ones( (self.data[0].shape[1]+(2*self.ref), self.data[0].shape[0]+(2*self.ref)), dtype=np.float32 )
        self.img_anchored = self.data[self.anchored]        
        self.canvas_anchored[self.ref:self.ref+self.data[0].shape[1],
                             self.ref:self.ref+self.data[0].shape[0]] = self.img_anchored        

    def createCanvasMovable(self):
        self.canvas_movable = np.ones( (self.data[0].shape[1]+(2*self.ref), self.data[0].shape[0]+(2*self.ref)), dtype=np.float32 )
        self.img_movable = self.data[self.movable]        
        self.canvas_movable[self.ref:self.ref+self.data[0].shape[1],
                            self.ref:self.ref+self.data[0].shape[0]] = self.img_movable       
    
    def refreshView(self):
        operation = str(self.combo_operation.currentText()).lower()
        self.img_align.setImage(self.operations(self.canvas_anchored, self.canvas_movable, operation), autoLevels=False)              
        self.plt_align.autoRange()
        print self.hist.getLevels()
        self.hist.autoHistogramRange()
        LUT = self.hist.getLookupTable(self.img_align.image)
        print type(LUT), LUT.shape
        img_hist = self.img_align.getHistogram()
        print type(img_hist), img_hist[0].shape, img_hist[1].shape
        print img_hist[0].max(), img_hist[0].min()
        print img_hist[1].max(), img_hist[1].min()
        self.hist.setLevels(img_hist[0].min(), img_hist[0].max())


    def alignImage(self):
        operation = str(self.combo_operation.currentText()).lower()
##        if operation != 'gk':
##            self.refreshView()
        self.canvas_movable[...] = 1
        self.canvas_movable[self.ref+self.spin_position_vertical.value():
                            self.ref+self.spin_position_vertical.value()+self.data[0].shape[1],
                            self.ref+self.spin_position_horizontal.value():
                            self.ref+self.spin_position_horizontal.value()+self.data[0].shape[0]] = self.img_movable
        self.img_align.setImage(self.operations(self.canvas_anchored, self.canvas_movable, operation), autoLevels=False)
        self.updatePosition()        

    def updatePosition(self):
        self.rel_pos[self.movable, self.anchored] = [self.spin_position_horizontal.value(),
                                                     self.spin_position_vertical.value()]
##        item = QtGui.QTableWidgetItem(str(self.rel_pos[self.movable, self.anchored]))
        index = self.tablePositions.currentIndex()
        item = widgets.TableWidget.TableWidgetItem( str(self.rel_pos[self.movable, self.anchored]), index )                    
        self.tablePositions.setItem(self.movable, self.anchored, item)
        
        self.tablePositions.setHorizontalHeaderLabels(self.hor_ver_header)
        self.tablePositions.setVerticalHeaderLabels(self.hor_ver_header)
        self.tablePositions.resizeColumnsToContents()
##        print self.rel_pos[self.movable]                   

    def operations(self, img0, img1, operation="multiplication", w1=1, w2=1):

        """Accepts two required arguments (img0, img1)
        and three optional arguments (operation, w1, w2)
        
        operation: string (multiplication, division, addition, subtraction)
        w1: float 0-1 
        w2: float 0-1
        """    

        if operation == "multiplication":
            return (w1*img0) * (w2*img1)        
        elif operation == "division":
            return (w1*img0) / (w2*img1)        
        elif operation == "addition":
            return img0 + img1        
        elif operation == "subtraction":
            return img0 - img1
        elif operation == "mask":            
            if self.mask.all() == 1:
                print "Please load a mask to perform this operation!"
                reply = QtGui.QMessageBox.information(self.winMain, "Warning",
"""Please load a mask to perform this operation!
""")
                self.openMask()                
            else:
                img_movable_masked = self.img_movable.copy() * self.mask 
                img_anchored_masked = self.img_anchored.copy() * self.mask
                            
##                s = np.zeros_like(img0)
                s = np.ones_like(img0)
    ##            print "s shape 1", s.shape
                s[self.ref:self.ref+self.data[0].shape[1],
                  self.ref:self.ref+self.data[0].shape[0]] = 1*self.mask
    ##            print "s shape 2", s.shape
                
                img0[self.ref:self.ref+self.data[0].shape[1],
                     self.ref:self.ref+self.data[0].shape[0]] = img_anchored_masked

                img1[self.ref+self.spin_position_vertical.value():
                     self.ref+self.spin_position_vertical.value()+self.data[0].shape[1],
                     self.ref+self.spin_position_horizontal.value():
                     self.ref+self.spin_position_horizontal.value()+self.data[0].shape[0]] = img_movable_masked
     
    ##            print img0.shape, img1.shape, s.shape
##                return (img0 + img1) / s
##                return (img0 + img1)
                return (img0 + img1) / s
            
    def get_xy_pos(self):
        x_pos = np.zeros( (self.row*self.col,) )
        y_pos = np.zeros( (self.row*self.col,) )

        x_pos[0] = self.rel_pos[0,0][0]
        y_pos[0] = self.rel_pos[0,0][1]

        for i in range(self.row*self.col):            
            if i == 0:
                x_pos[i] = self.rel_pos[i,i][0]
                y_pos[i] = self.rel_pos[i,i][1]
            elif (i%self.row) == 0:
                x_pos[i] = self.rel_pos[i, i-self.row][0] + x_pos[i-self.row]
                y_pos[i] = self.rel_pos[i, i-self.row][1] + y_pos[i-self.row]                    
            else:
                x_pos[i] = self.rel_pos[i, i-1][0] + x_pos[i-1]
                y_pos[i] = self.rel_pos[i, i-1][1] + y_pos[i-1]       

        return x_pos, y_pos
##        return y_pos, x_pos

    def previewRec(self):
        # get absolute x and y positions
        x_pos, y_pos = self.get_xy_pos()
        print x_pos
        print y_pos
##        x_pos = np.asarray([0,0,0,-100,-100,-100,-200,-200,-200])
##        y_pos = np.asarray([0,-100,-200,0,-100,-200,0,-100,-200])
        #x y pos sanitizing
        x_pos = np.abs(x_pos)
        y_pos = np.abs(y_pos)

        x_pos -= x_pos.min()
        y_pos -= y_pos.min()

        x_pos = x_pos.astype(int)
        y_pos = y_pos.astype(int)

        xmax = np.asarray( [x_pos[i] + self.data[i].shape[0] for i in range( len(self.probes_paths[0]) )] ).max()
        ymax = np.asarray( [y_pos[i] + self.data[i].shape[1] for i in range( len(self.probes_paths[0]) )] ).max()

        c = np.zeros( shape=(xmax,ymax), dtype=np.float32 )

        s = np.zeros_like( c ) #sample density

        for i in range( len(self.probes_paths[0]) ):
            print '.',
            c[ x_pos[i]:x_pos[i]+self.data[i].shape[0],
               y_pos[i]:y_pos[i]+self.data[i].shape[1] ] +=  np.rot90( np.flipud(self.data[i]), k=1 ) * np.rot90(self.mask, k=1) #d[i]*m is dm[i]
            s[ x_pos[i]:x_pos[i]+self.data[i].shape[0], y_pos[i]:y_pos[i]+self.data[i].shape[1] ] += np.rot90(self.mask, k=1)
##            c[ x_pos[i]:x_pos[i]+self.data[i].shape[0],
##               y_pos[i]:y_pos[i]+self.data[i].shape[1] ] +=  np.rot90( np.flipud(self.data[i]) , k=1 ) * np.rot90(self.mask, k=1) #d[i]*m is dm[i]
##            s[ x_pos[i]:x_pos[i]+self.data[i].shape[0], y_pos[i]:y_pos[i]+self.data[i].shape[1] ] += np.rot90(self.mask, k=1)

        wc = c/s
        wc = np.rot90(wc, k=3)

        self.winPreview.setWindowTitle('Global Image - mask:%s' % self.maskname)

        self.imvPreview.setImage(wc)

        self.winPreview.show()

        print 'done.'

        
##        # first idea: lunch a separate script
##        tf = tempfile.NamedTemporaryFile(suffix=".npy")
##        tf = tf.name
##        print tf
####        1/0
##        np.save(tf, self.rel_pos)
##        prog = "python ./FGpty/scripts/AlignmentEditorCarlos.py"
##        command = "(%s %s; rm -f %s)&" % (prog, tf, tf)
##        os.system(prog)
##        time.sleep(0.5)    

                

        
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        gui = PtychoAlign()
        gui.winMain.show()
        QtGui.QApplication.instance().exec_()

##    np.save("rel_pos", window.rel_pos)

        
