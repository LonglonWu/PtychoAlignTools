import sys
import skimage
from skimage.filters import threshold_otsu, threshold_adaptive
print ("skimage v.",skimage.__version__)
from skimage import data, io, filters, morphology, feature, util
import numpy as np
print("numpy v.",np.__version__)
import tifffile
print('tifffile v.',tifffile.__version__)
import matplotlib as plt
from matplotlib import cm
#import matplotlib.pyplot as plt
print("matplotlib v.",plt.__version__)
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pylab as plb
#print('pylab v.', plb.__version__)
import scipy
import scipy.optimize
print('scipy v.', scipy.__version__)
from scipy import asarray as ar,exp

import seaborn as sns
#import progressbar
from mpl_toolkits.mplot3d import Axes3D

# Load test and check alignment
from LoadDataTest import LoadData,LoadData2
from eqmeas_lib import doOneImageTest,normalizeAll,MyPlot3d,displayStatsGen,addDescrption
#from eqmeas3_xy import jitterAdd,alignImage,normalizeAll,displayStats,surface_plot


################################################################################

###################################################################################
###
# start
testTable = LoadData2(rot90=True)
XMOV = int(3)
YMOV = int(3)
mask = testTable[0].astype(np.float32)

mask -= mask.min()
mask /= mask.max()
io.imshow(mask)
io.show()
MovRangeY = np.arange(-YMOV,YMOV+1)
MovRangeX = np.arange(-XMOV,XMOV+1)
outPicPath = 'outTestEqMeas/'
x, y = np.meshgrid(MovRangeY, MovRangeX)

for testN in range(1,len(testTable),1):
    print('Test:',testN)
    testPar = testTable[testN]
    resTable = doOneImageTest(mask=mask,
                       fn=testPar[0],
                       x_pos=testPar[1],
                       y_pos=testPar[2],
                       MovRangeX=MovRangeX,
                       MovRangeY=MovRangeY)

    Mse = resTable[0]
    Sad = resTable[1]
    PwStd = resTable[2]
    Xcorr = resTable[3]
    CorrRatio = resTable[4]
    print(CorrRatio.shape)
    #CorrRatio = resTable[0]

    '''
    io.imsave(outPicPath + 'Mse_test_' + str(testN) + '.png', Mse)
    io.imsave(outPicPath + 'Sad_test' + str(testN) + '.png', Sad)
    io.imsave(outPicPath + 'PwStd_test_' + str(testN) + '.png', PwStd)
    '''
    
#    MyPlot3d(Sad,x,y,
#             title='Sad_test_3d_' + str(testN) + '.png',
#             save=True,setOnTop=False)
#    
#    MyPlot3d(Mse,x,y,
#             title='Mse_test_3d_' + str(testN) + '.png',
#             save=True,setOnTop=False)
#        
#    MyPlot3d(PwStd,x,y,
#             title='PwStd_test_3d_' + str(testN) + '.png',
#             save=True,setOnTop=False)
    manySurface = np.asarray([Sad,CorrRatio])
    manySurface = normalizeAll(manySurface,size=1)
    #MyPlot3d(np.asarray([CorrRatio,Mse]),x,y,
    MyPlot3d(manySurface,x,y,
             title='CorrRatio_3d_' + str(testN) + '.png',
             save=True,setOnTop=False)
    
    #2d plot in 050
    plottable = []
    plottable.append(Mse[XMOV,:])
    plottable.append(Sad[XMOV,:])
    plottable.append(PwStd[XMOV,:])
    plottable.append(CorrRatio[XMOV,:])
    
    plottable = normalizeAll(plottable,subMin=True)
    descr = ['Mse x0','Sad x0','Pwstd x0', 'CorrRatio x0']
    #descr = ['CorrRatio x0']
    plottable = plottable.tolist()
    plottable.append(descr)
    displayStatsGen(MovRangeY,plottable,title='Test:' + str(testN) + ' Stats in x = 0')
    
    plottable = []
    plottable.append(Mse[:,YMOV])
    plottable.append(Sad[:,YMOV])
    plottable.append(PwStd[:,YMOV])
    plottable.append(CorrRatio[:,YMOV])
    
    plottable = normalizeAll(plottable,subMin=True)
    plottable = plottable.tolist()
    descr = ['Mse y0','Sad y0','Pwstd y0', 'CorrRatio y0']
    #descr = ['CorrRatio y0']
    plottable.append(descr)
    displayStatsGen(MovRangeX,plottable,title='Test:' + str(testN) + ' Stats in y = 0')


    meanOnY_Mse = Mse.mean(axis=0)
    #meanOnY_MeanDiff = MeanDiff.mean(axis=0)
    meanOnY_Sad = Sad.mean(axis=0)
    meanOnY_Xcorr = Xcorr.mean(axis=0)
    meanOnY_PwStd = PwStd.mean(axis=0)
    meanOnY_CorrRatio = CorrRatio.mean(axis=0)
    plottable = []
    #plottable.append(meanOnY_MeanDiff)
    plottable.append(meanOnY_Mse)
    plottable.append(meanOnY_Sad)
    plottable.append(meanOnY_Xcorr)
    plottable.append(meanOnY_PwStd)
    plottable.append(meanOnY_CorrRatio)
    plottable = normalizeAll(plottable,subMin=True)
    plottable = addDescrption(plottable, ['MeanOnY_Mse','MeanonY_Sad','MeanOnY_XCorr','MeanOnY_PwStd', 'meanOnY_CorrRatio'])
    #plottable = addDescrption(plottable, ['meanOnY_CorrRatio'])
    displayStatsGen(MovRangeY,plottable,title='Average on Axis 0 - Test:' + str(testN)) #Y

    meanOnX_Mse = Mse.mean(axis=1)
    #meanOnX_MeanDiff = MeanDiff.mean(axis=1)
    meanOnX_Sad = Sad.mean(axis=1)
    meanOnX_Xcorr = Xcorr.mean(axis=1)
    meanOnX_PwStd = PwStd.mean(axis=1)
    meanOnX_CorrRatio = CorrRatio.mean(axis=1)
    plottable = []
    #plottable.append(meanOnX_MeanDiff)
    plottable.append(meanOnX_Mse)
    plottable.append(meanOnX_Sad)
    plottable.append(meanOnX_Xcorr)
    plottable.append(meanOnX_PwStd)
    plottable.append(meanOnX_CorrRatio)
    plottable = normalizeAll(plottable,subMin=True)
    plottable = addDescrption(plottable, ['MeanOnX_Mse','MeanonX_Sad','MeanOnX_XCorr','MeanOnX_PwStd','meanOnX_CorrRatio'])
#    plottable = addDescrption(plottable, ['meanOnX_CorrRatio'])
    displayStatsGen(MovRangeX,plottable,title='Average on Axis 1 - Test:' + str(testN)) #X

#    idxMinY = np.argmin(meanOnY_PwStd) #y vector
#    idxMinX = np.argmin(meanOnX_PwStd) #x vector
    
    idxMinY = np.argmax(meanOnY_CorrRatio) #y vector
    idxMinX = np.argmax(meanOnX_CorrRatio) #x vector

    tx = MovRangeX[idxMinX]
    ty = MovRangeY[idxMinY]
    print('New pos is:' + str(testPar[1][1] + tx) + ';' + str(testPar[2][1] + ty))
    
    #saving data
    print('Saving data')
    np.save("outTestEqMeas/test_" + str(testN) + ".npy", resTable)
    
    print('done test')
