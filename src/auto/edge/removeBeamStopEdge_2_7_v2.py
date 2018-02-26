
# coding: utf-8

# In[1]:


#remove beamstop edge from edges in every image
from __future__ import print_function
import skimage 
print ("skimage v.",skimage.__version__)
from skimage import data, io, filters, morphology, feature, util
import numpy as np
#import numpy.ma as ma
print("numpy v.",np.__version__)
#import theano
#print("Theano v.",theano.__version__)
#import torch
#print("torch v.",torch.__ver sion__)
#import cv2
#print("OpenCv v.",cv2.__version__)
import tifffile as tf
print('tifffile v.',tf.__version__)
get_ipython().magic(u'matplotlib inline')
import matplotlib as plt
#import matplotlib.pyplot as plt
print("matplotlib v.",plt.__version__)
from skimage.filters import threshold_otsu, threshold_adaptive

basePath = "/home/francesco/Seafile/PHD/imageAlign/testPy/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y"
trailerPath = ".000000-x0.000000.tiff"  
#LoadedImage = np.zeros([12,2048,2048])
beamstopEdge = np.flipud(io.imread("/home/francesco/Seafile/PHD/imageAlign/testPy/dilatedborder.png"))
#beamstopEdge = skimage.util.invert(beamstopEdge)
print("BeamStop max:",np.amax(beamstopEdge),"BeamStop min:",np.amin(beamstopEdge))
#beamstopEdge = beamstopEdge.astype(np.bool)
#beamstopEdge = np.amax(beamstopEdge) - beamstopEdge
beamstopEdge = filters.gaussian(beamstopEdge, sigma=10)
print('BeamStopEdge')

beamstopEdge = beamstopEdge > 0.1 #np.amax(beamstopEdge)
#fig = plt.pyplot.figure(figsize=(16, 6))
#ax1 = plt.pyplot.subplot(1,2,1)
#ax1.imshow(beamstopEdge,cmap='gray')
#io.imshow(beamstopEdge)
#io.show()
#print("max:",np.amax(beamstopEdge),"min:",np.amin(beamstopEdge))
#beamstopEdge = beamstopEdge.astype(np.bool)
#beamstopEdge = True*beamstopEdge

print("Loading...")
for i in range(12):
    print("Loading: " + str(i))
    imgCompletePath = basePath+str(i)+trailerPath
    #LoadedImage[i] = np.flipud(tf.imread(imgCompletePath))
    LoadedImage = (tf.imread(imgCompletePath))
    border = filters.sobel(LoadedImage)
    #print("sobel")
    #io.imshow(LoadedImage)
    #io.show()
    #dilatedBorder = morphology.binary_dilation(border,morphology.square(5))
    #dilatedBorder = morphology.binary_dilation(dilatedBorder,morphology.square(5))
    #border = ma.array(border,mask=beamstopEdge)
    fig = plt.pyplot.figure(figsize=(16, 6))
    ax1 = plt.pyplot.subplot(1, 4, 1, adjustable='box-forced')
    ax2 = plt.pyplot.subplot(1, 4, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
    ax3 = plt.pyplot.subplot(1, 4, 3)
    ax4 = plt.pyplot.subplot(1, 4, 4)
        
    print("masking sobel..")
    border[beamstopEdge] = 0
    #io.imshow(border)
    #io.show()
    
    ax1.imshow(border, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Masked Sobel')
    
    print("opening")
    border = morphology.opening(border,morphology.diamond(3))
    #io.imshow(border)
    #io.show()
    
    ax2.imshow(border, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Opening')
    
    border = border/np.amax(border)
    border = border**2.7
    print("power law norm")
    #io.imshow(border)
    #io.show()
    
    ax3.imshow(border, cmap='gray')
    ax3.set_axis_off()
    ax3.set_title('Power law')
    
    #print("border max:",np.amax(border),"borer min:",np.amin(border))
    #print("histogram")
    #histBorder,histBin = np.histogram(border,bins=1000)
    #plt.pyplot.plot(histBin[1:],histBorder)
    #plt.pyplot.hist(border, bins=10)
    
    print("thresholded and morphological")

    #border = morphology.closing(border,morphology.square(5))
    whiteValMask = border > 0.07
    border[whiteValMask] = 1
    whiteValMask = border < 0.07
    border[whiteValMask] = 0
    
    border = morphology.binary_closing(border,morphology.diamond(1))
    border = morphology.binary_closing(border,morphology.diamond(1))
    #border = morphology.skeletonize(border)
    border = morphology.binary_erosion(border,morphology.diamond(1))
    #border = skimage.filters.rank.median(skimage.morphology.disk(5))
    #border = morphology.erosion(border,morphology.square(5))
    
    ax4.imshow(border, cmap='gray')
    ax4.set_axis_off()
    ax4.set_title('thresholded and morph')
    #plt.pyplot.show()
    
    #border = feature.canny(border)

    #io.imshow(border)
    #io.show()
    io.imsave("output_feat_lines/featExt"+str(i+1)+".png",255*border)
    plt.pyplot.savefig("output_feat_lines_proc/featExtProcess"+str(i+1)+".png", dpi=1200)
    display(fig)
print("end")

