####EQ MEAS LIBS
import sys
import skimage
from skimage.filters import threshold_otsu, threshold_adaptive
#print ("skimage v.",skimage.__version__)
from skimage import data, io, filters, morphology, feature, util
import numpy as np
#print("numpy v.",np.__version__)
import tifffile
#print('tifffile v.',tifffile.__version__)
import matplotlib as plt
from matplotlib import cm
#import matplotlib.pyplot as plt
#print("matplotlib v.",plt.__version__)
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pylab as plb
#print('pylab v.', plb.__version__)
import scipy
import scipy.optimize
#print('scipy v.', scipy.__version__)
from scipy import asarray as ar,exp

import seaborn as sns
#import progressbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
# Load test and check alignment
#from LoadDataTest import LoadData

outPath = 'outTestEqMeas/'

fontP = FontProperties()
fontP.set_size('medium')

def jitterAdd( seq, jitter ):
    r = []
    r.append( seq[0] )
    for i in range( 1, len(seq) ):
         v = r[-1]    +    (abs(seq[i] - seq[i-1]) + jitter)
         r.append( v )
    return np.asarray( r ).astype( int )


def alignImage(d, xp, yp, m, MinSampleDens=2):
    yp -= yp.min()
    xp -= xp.min()
    xmax = np.asarray( [xp[i] + d[i].shape[0] for i in range( d.shape[0] )] ).max()
    ymax = np.asarray( [yp[i] + d[i].shape[1] for i in range( d.shape[0] )] ).max()

    imc = np.zeros((2,xmax,ymax),dtype=np.float32) #stacked img canvas
    s = np.zeros( shape=(xmax,ymax), dtype=np.float32 ) #sample density

    for i in range( d.shape[0] ):
        #print ('.'),
        s[ xp[i]:xp[i]+d[i].shape[0], yp[i]:yp[i]+d[i].shape[1] ] += m
        imc[i][ xp[i]:xp[i]+d[i].shape[0], yp[i]:yp[i]+d[i].shape[1] ] = d[i] * m

    imc[:,s<MinSampleDens] = 0
    return imc

'''
def normalizeAll(data, subMin=True):
    data = np.asanyarray(data,dtype=np.float32)
    for i in range(data.shape[0]):
        if subMin:
            data[i] -= np.amin(data[i])
        data[i] /= np.amax(data[i])
    return data
'''

def normalizeAll(data, subMin=True, size=1):
    if size == 1:
        data = np.asanyarray(data,dtype=np.float32)
        for i in range(data.shape[0]):
            if subMin:
                data[i] -= np.amin(data[i])
            data[i] /= np.amax(data[i])
    else:
        if subMin:
            data -= np.amin(data)
        data /= np.amax(data)
    return data

'''
def displayStats(MovRange,data,title='Default'):
    fig = plt.pyplot.figure(figsize=(12, 6))
    axHDiff = plt.pyplot.subplot(1, 2, 1, adjustable='box-forced')
    if len(data) == 5:
        axHDiff.plot(MovRange, data[0], '+:',label='Mean of abs(diff)')
        axHDiff.plot(MovRange, data[1], '+:', label='MSE')
        axHDiff.plot(MovRange, data[2], '+:', label='SAD')
        axHDiff.plot(MovRange, data[3], '+:', label='xCorr')
        axHDiff.plot(MovRange, data[4], '+:', label='Mean of pw std')
    else:
        axHDiff.plot(MovRange, data[0], '+:', label='MSE')
        axHDiff.plot(MovRange, data[1], '+:', label='SAD')
        axHDiff.plot(MovRange, data[2], '+:', label='xCorr')
        axHDiff.plot(MovRange, data[3], '+:', label='Mean of pw std')
        axHDiff.grid(color='k')

    axHDiff.legend()
    if title== 'Default':
        axHDiff.set_title('Translation dependent Statistics')
    else:
        axHDiff.set_title(title)
    axHDiff.set_xlabel('Translation')
    axHDiff.set_ylabel('Value')
    axHDiff.set_xticks(MovRange)
'''

def displayStatsGen(MovRange, dataTable, title='', save=True):
    graphDescr = dataTable[len(dataTable)-1]
    assert(len(dataTable) -1 == len(graphDescr))
    fig = plt.pyplot.figure(figsize=(12, 6))
    axHDiff = plt.pyplot.subplot(1, 2, 1, adjustable='box-forced')
    for i in range(len(graphDescr)):
        axHDiff.plot(MovRange,dataTable[i],label=graphDescr[i])
    #axHDiff.legend()
    axHDiff.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)
    axHDiff.set_xlabel('Translation')
    axHDiff.set_ylabel('Value')
    axHDiff.grid(color='k')
    if title != '':
        axHDiff.set_title(title)
    else:
        save = False
    if MovRange.shape[0] < 20:     
        axHDiff.set_xticks(MovRange)
    if save:
        fig.savefig(outPath + title + '.png', bbox_inches='tight',dpi=900)

def surface_plot (matrix,x,y, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    #(x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    colormaps = ['winter','autumn','spring', 'cool', 'Greens', 'Purples', 'Blues', 'Greys', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    colormaps = ['red','blue']
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(matrix.shape) == 3:
        for i in range(matrix.shape[0]):
            #surf = ax.plot_surface(x, y, matrix[i], alpha=0.35, rstride=1, cstride=1, cmap=colormaps[i])
            surf = ax.plot_wireframe(x, y, matrix[i], rstride=1, cstride=1, color=colormaps[i])
            #surf = ax.contour(x, y, matrix[i], zdir='x', offset=0, cmap=colormaps[i])
            #surf = ax.contour(x, y, matrix[i], zdir='y', offset=0, cmap=colormaps[i])
    else:
        surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)
        
#def surface_plot (matrix,x,y, **kwargs):
#    # acquire the cartesian coordinate matrices from the matrix
#    # x is cols, y is rows
#    #(x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
#    fig = plt.pyplot.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    if len(matrix.shape) == 3:
#        colormaps = ['winter','autumn','spring', 'cool', 'Greens', 'Purples', 'Blues', 'Greys', 'Oranges', 'Reds',
#            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#        for i in range(matrix.shape[0]):
#            surf = ax.plot_surface(x, y, matrix[i], cmap=colormaps[i],rstride=8, cstride=8, alpha=0.35+i/10)
            #ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#            cset = ax.contour(X, Y, Z, zdir='z', offset=np.amin(Z), cmap=cm.coolwarm)
#            cset = ax.contour(X, Y, Z, zdir='x', offset=np.amin(x), cmap=cm.coolwarm)
#            cset = ax.contour(X, Y, Z, zdir='y', offset=np.amax(y), cmap=cm.coolwarm)
            #surf = ax.contour(x, y, matrix[i], zdir='z', offset=np.amin(matrix[i]), cmap=colormaps[i])
#            surf = ax.contour(x, y, matrix[i], zdir='x', offset=np.amin(x), cmap=colormaps[i])
#            surf = ax.contour(x, y, matrix[i], zdir='y', offset=np.amax(y), cmap=colormaps[i])
#    else:
#        print(matrix.shape)
#        surf = ax.plot_surface(x, y, matrix, **kwargs)
#    return (fig, ax, surf)

def MyPlot3d(data2d,x,y,title,save=True,setOnTop=True):
    (fig, ax, surf) = surface_plot(data2d,x,y, cmap=plt.pyplot.cm.coolwarm)
    #fig.colorbar(surf)
    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    ax.set_title(title)
    if setOnTop:
        ax.view_init(elev=-90., azim=90)
    if save:
        fig.savefig(outPath + title, bbox_inches='tight',dpi=900)
    plt.pyplot.show()

def moveXorY(xp,yp,jitter,XorY='X'):
    if XorY == 'X':
        xp = jitterAdd(xp, jitter)
    else:
        yp = jitterAdd(xp, jitter)
    return xp,yp


def calcCorrRatio(img,binN=32):
    img0 = img[0]
    img1 = img[1]
    # quantizza immagine in interi
    img0 = np.round((binN-1)*img0).astype(np.int)
    img1 = np.round((binN-1)*img1).astype(np.int)
    #io.imshow(img0)
    #io.show()

    imgs = int(img0.size)
#    if np.ma.is_masked(img0) and np.ma.is_masked(img1):
#        img0 = img0.compressed()
#        img1 = img1.compressed()
    hist,bin_edg = np.histogram(img0,bins=np.arange(1,binN)) # non contare lo zero
    #print(hist)
    #hist[0] = int(0)
    imgs = int(hist.sum())
    hist = hist/imgs

    vari = np.zeros(hist.size) + (2*(binN-1))**2 # set variance to the max
    ##vari[0] = 0

    for i in range(hist.size):
        xiPos = (img0 == i)
        if xiPos.sum() > 0:
            vari[i] = img1[xiPos].var()

    Dsq = (hist*vari).sum()/hist.sum() #this is dsquare
    Dsq = Dsq/img1[img1!=0].var()
    #Dsq = Dsq/img1.var()
    #print(Dsq)
    return np.sqrt(1-Dsq)
    #return Dsq


def doOneImageTest(mask,fn,x_pos,y_pos,MovRangeX,MovRangeY,rot90=True):
    COEFF = int(2**8)
    d = np.asarray([ np.rot90(tifffile.imread( f )) for f in fn])
    d[0] = d[0] * mask
    d[1] = d[1] * mask
    
    #crop images
    d = np.asarray([d[f,500:1600,500:1600] for f in range(d.shape[0])])
    mask = mask[500:1600,500:1600]

    #count = 0
    Sad = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)
    Mse = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)
    PwStd = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)
    MeanDiff = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)
    Xcorr = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)
    CorrRatioMasked = np.zeros((MovRangeX.size,MovRangeY.size),dtype=np.float)

    #bar = progressbar.ProgressBar(max_value=MovRangeX.size*MovRangeY.size)
    print('Generating a ' + str(MovRangeX.size) + 'x' + str(MovRangeY.size) + ' matrix')
    countX = 0
    for jx in MovRangeX:
        countY = 0
        #print('line:',countX)
        for jy in MovRangeY:
            #bar.update(count)
            sys.stdout.write('.'), sys.stdout.flush()
            xp = x_pos.copy()
            yp = y_pos.copy()

            #xp,yp = moveXorY(xp,yp,jy,'Y')
            xp = jitterAdd(xp,jx)
            yp = jitterAdd(yp,jy)

            yp -= yp.min()
            xp -= xp.min()

            imc = alignImage(d, xp, yp, mask)
            diffImg = np.abs(imc[0] - imc[1])
            prodImg = imc[0] * imc[1]

            Mse[countX][countY] = np.sum(np.power(diffImg,2)) #MSE
            Sad[countX][countY] = np.sum(diffImg)
            MeanDiff[countX][countY] = np.mean(diffImg)
            PwStd[countX][countY] = (imc.std(axis=0)).mean()
            Xcorr[countX][countY] = np.sum(prodImg)
            CorrRatioMasked[countX][countY] = calcCorrRatio(imc, binN=COEFF)

            '''
            if jx == 0 or jy == 0:
                resTable = []
                resTable.append(MeanDiff[countX])
                resTable.append(Mse[countX])
                resTable.append(Sad[countX]) #SAD
                resTable.append(Xcorr[countX])
                resTable.append(PwStd[countX])
                resTable = normalizeAll(resTable,subMin=True)
                displayStats(MovRangeY,resTable)
            '''
            #count += 1 #total counter
            countY +=1
        print('\n')
        countX += 1

    resTable = []
    resTable.append(Mse)
    resTable.append(Sad)
    resTable.append(PwStd)
    resTable.append(Xcorr)
    resTable.append(CorrRatioMasked)
    io.imshow(CorrRatioMasked)
    io.show()
    return resTable

def addDescrption(plottable, descr):
    plottable = plottable.tolist()
    plottable.append(descr)
    return plottable
