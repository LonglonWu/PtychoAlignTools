from __future__ import print_function
import skimage
from skimage.filters import threshold_otsu, threshold_adaptive
print ("skimage v.",skimage.__version__)
from skimage import data, io, filters, morphology, feature, util, img_as_float, img_as_bool
import numpy as np
print("numpy v.",np.__version__)
import tifffile as tf
print('tifffile v.',tf.__version__)
import matplotlib as plt
#import matplotlib.pyplot as plt
print("matplotlib v.",plt.__version__)
import pylab as plb
#print('pylab v.', plb.__version__)
import scipy
import scipy.optimize
print('scipy v.', scipy.__version__)
from scipy import asarray as ar,exp

#import warnings
#warnings.filterwarnings("ignore")

#user Vars
basePath = "/home/francesco/Seafile/PHD/imageAlign/testPy/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/"
baseFileName = "mag_778.0eV_0288.00mu_fd0.3050000-y"
trailerPath = ".000000-x0.000000.tiff"
basePathPos = "/home/francesco/PHDelettra/imageAlign/testPy/overlapped/"
basePathImg = "/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y"
trailerPath = ".000000-x0.000000.tiff"
BeamStopMaskPath = "/home/francesco/Seafile/FGpty/fra_2-3/ilum2.tif"

IMGNUM = 10
#img = skimage.util.img_as_float(skimage.data.camera())
img = tf.imread(basePath + baseFileName + str(IMGNUM) + trailerPath)
img = img[500:1600,500:1600]
#io.imshow(img)
#io.show()
histImg = np.asanyarray(skimage.exposure.histogram(img, nbins=2**8))
print(histImg.size)
#plt.pyplot.title('Orig Hist')
#plt.savefig(OutputFolderPath + "Algn_" + str(imgNum1) + "_" + str(imgNum2) + "__" + str(posX[1]) + "__" + str(posY[1]) + ".png")
#display(fig3)
#plt.close(fig3)
#plt.pyplot.plot(histImg[1],histImg[0],'b+:',label='orig. his')
MaskValue = skimage.util.invert(tf.imread(BeamStopMaskPath))
MaskValue = MaskValue[500:1600,500:1600]
#io.imshow(MaskValue)
#io.show()
MaskValue = skimage.morphology.binary_dilation(MaskValue,skimage.morphology.square(11))
MaskValue = MaskValue > 0

MaskValue = skimage.util.img_as_bool(MaskValue)
MaskValueOrig = MaskValue.copy()

img[MaskValueOrig] = 0

# cut histogram black and white
THRSL = 0.2
THRSH = 0.7
for k in range(histImg[1].size):
    histImg[0][k] = histImg[0][k] * \
                    (histImg[1][k] > THRSL and histImg[1][k] < THRSH).astype(np.float)
# fix area
histImg[0] /= np.sum(histImg[0].ravel())

'''
plt.pyplot.plot(histImg[1] ,histImg[0],'r+:',label='mod hist')
plt.pyplot.legend()
plt.pyplot.title('Fig. 1, Modified histogram')
plt.pyplot.xlabel('Level')
plt.pyplot.ylabel('Prob')
plt.pyplot.show()
'''




# histogram curve fitting (gaussian or bi-gaussian)
def bigaus(x, a, b, x0, stdA, stdB):
    partA = a*exp(-(x-x0)**2/(2*stdA**2))*(x<=x0)
    partB = b*exp(-(x-x0)**2/(2*stdB**2))*(x>x0)
    return (partA + partB)/(a+b)

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

# detect min and max value
def getIdxOfPercentile(array, perc):
    return round((len(array) - 1) * (perc / 100.0))/array.size

def cutHeadTailHist(image, THRSL=0.2, THRSH=0.7, nbins=2**8, type='gaus', plot=False):
    # histogram curve fitting (gaussian or bi-gaussian)

    histImg = np.asanyarray(skimage.exposure.histogram(img, nbins=nbins))
    if plot:
        plt.pyplot.plot(histImg[1] ,histImg[0],'r+:',label='mod hist')
        plt.pyplot.legend()
        plt.pyplot.title('Orig histogram')
        plt.pyplot.xlabel('Level')
        plt.pyplot.ylabel('Prob')
        plt.pyplot.show()
    histImg[0] = histImg[0] * (histImg[1] > THRSL and histImg[1] < THRSH).astype(np.float)
    # fix area
    histImg[0] /= np.sum(histImg[0].ravel())
    if plot:
        plt.pyplot.plot(histImg[1] ,histImg[0],'r+:',label='mod hist')
        plt.pyplot.legend()
        plt.pyplot.title('Modified histogram')
        plt.pyplot.xlabel('Level')
        plt.pyplot.ylabel('Prob')
        plt.pyplot.show()
    x = histImg[1]
    y = histImg[0]
    popt,pcov = scipy.optimize.curve_fit(gaus,x,y)
    fitValues = gaus(x,*popt)
    LowVal = getIdxOfPercentile(fitValues, 15)
    HighVal = getIdxOfPercentile(fitValues, 70)
    print('Detected vals:', LowVal,HighVal)
    # threshold out-window values
    mask0 = img < LowVal
    mask1 = img > HighVal
    imgNew = img.copy()
    imgNew[mask0] = 0
    imgNew[mask1] = 1
    return imgNew

#NCOST = 2500
x = histImg[1]
y = histImg[0]
#minArr = scipy.signal.argrelextrema(scipy.signal.medfilt(y.copy(),3), np.less)[0]
#meanValue = np.argmax(y) #pdf-mod peak
#popt,pcov = scipy.optimize.curve_fit(bigaus,x,y, p0=[0.1, 0.1, meanValue, 1 ,1])
popt,pcov = scipy.optimize.curve_fit(gaus,x,y)
#fitValues = bigaus(x,*popt)*NCOST
fitValues = gaus(x,*popt)
#distribution normalization
#DistrArea = fitValues.ravel().sum()
#fitValues /= DistrArea

'''
# plot resulting histogram and fit
plt.pyplot.plot(x,y,'b+:',label='histogram')
plt.pyplot.plot(x, fitValues,'r+:',label='fit')
plt.pyplot.legend()
plt.pyplot.title('Fig. 2 - gaussian fit')
plt.pyplot.xlabel('Level')
plt.pyplot.ylabel('Prob.')
plt.pyplot.show()
'''
#Descr = scipy.stats.describe(fitValues)
#print(Descr)

LowVal = getIdxOfPercentile(fitValues, 15)
HighVal = getIdxOfPercentile(fitValues, 70)
#LowVal = THRSL #minArr[0]
#HighVal = THRSH #minArr[1]
print('Detected vals:', LowVal,HighVal)

# threshold out-window values
#mask = np.logical_and(img < LowVal,img > HighVal)
mask0 = img < LowVal
mask1 = img > HighVal
'''
imgCopy = img.copy()
imgCopy[mask0] = 0
imgCopy[mask1] = 0
meanValAllImg = np.mean(imgCopy)
'''
imgNew = img.copy()
imgNew[mask0] = 0
imgNew[mask1] = 1

'''
def mapData(x, low, high):
    temp = np.zeros_like(x)
    temp =  (x-low)/(high-low)
    return temp

imgNew = mapData(imgCopy.copy(), low=LowVal, high=HighVal)
'''
# CLAHE, median filtering, mapping
imgNew = skimage.exposure.equalize_adapthist(imgNew, kernel_size=100,nbins=2**16)
imgNew = skimage.filters.rank.median(imgNew, morphology.diamond(21))
imgNew = imgNew**0.8
imgNew = imgNew - np.amin(imgNew)
imgNew = imgNew/np.amax(imgNew)

fig = plt.pyplot.figure(figsize=(12, 4))#,dpi=100)
ax1 = plt.pyplot.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.pyplot.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
ax3 = plt.pyplot.subplot(1, 3, 3, adjustable='box-forced')

ax1.imshow(img, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Original')

ax2.imshow(imgNew, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Modified')

# histogram of the modified image, remove masked value bin, fix area
imgNewHist = np.asanyarray(skimage.exposure.histogram(imgNew, nbins=2**10))
idxmaxNewH = np.argmax(imgNewHist[0])
imgNewHist[0][idxmaxNewH] = 0
imgNewHist[0] /= np.sum(imgNewHist[0].ravel())

#ax3.hist(imgNew.ravel(), bins=2**8, range=(0,1), fc='k', ec='k')
ax3.plot(imgNewHist[1],imgNewHist[0])
#ax3.set_ylim([0,1e5])
ax3.set_title('OutImage histogram')
plt.pyplot.show()

#extract features
imgNewTh = np.ones_like(imgNew,dtype=np.float)
imgNewTh = skimage.util.invert(imgNew)
imgNewTh[MaskValue] = 0
MaskimgNewTh = imgNewTh < 0.65
imgNewTh[MaskimgNewTh] = 0
imgNewTh[np.invert(MaskimgNewTh)] = 1
#imgNewTh = skimage.util.invert(imgNewTh)
#io.imshow(imgNewTh)
#io.show()

imgNewTh_removeBig = skimage.morphology.remove_small_objects(imgNewTh.astype(np.bool),2000,connectivity=8)
imgNewTh_removeSmall = skimage.morphology.remove_small_objects(imgNewTh.astype(np.bool),200,connectivity=8)
#imgNewTh_removeSmall = imgNewTh_removeSmall - imgNewTh_removeBig
io.imshow(imgNewTh_removeSmall)
io.show()

border = skimage.feature.canny(MaskValueOrig)
border = img_as_bool(border)

border = skimage.morphology.binary_dilation(border,skimage.morphology.square(27))
#io.imshow(border)
#io.show()

imgNewTh_removeSmall[border] = 0
#io.imshow(imgNewTh_removeSmall, cmap='hot')
#io.show()

'''
print('watershed')
local_maxi = skimage.feature.peak_local_max(imgNewTh, min_distance = 5)
xMaxPos = local_maxi[:,0]
yMaxPos = local_maxi[:,1]
LocalMaxiCanvas = np.zeros((2048,2048), dtype=np.float)
LocalMaxiCanvas[xMaxPos,yMaxPos] = 1
WaImg = skimage.morphology.watershed(imgNewTh,imgNewTh)#, markers=LocalMaxiCanvas, mask=np.invert(MaskValue))
io.imshow(WaImg)
io.show()
'''
#, indices=False, footprint=np.ones((3, 3)),                         labels=image)
#MaskValue = np.logical_or(MaskValue, ModImage2 > 0.15)
#ModImage2[MaskValue] = 1
#io.imshow(ModImage2)
#io.show()
#LoadedImage[Lidx] = np.ma.masked_array(ModImage2.copy(), mask=MaskValue)
#io.imshow(LoadedImage[Lidx])
#io.show()

fig = plt.pyplot.figure(figsize=(12, 4))#,dpi=100)
ax1 = plt.pyplot.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.pyplot.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
ax3 = plt.pyplot.subplot(1, 3, 3, adjustable='box-forced')

ax1.imshow(img, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Original image')

ax2.imshow(imgNew, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Processed image')

ax3.imshow(imgNewTh_removeSmall, cmap='gray')
ax3.set_axis_off()
ax3.set_title('Segmented features')







