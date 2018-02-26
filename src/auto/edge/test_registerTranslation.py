
# coding: utf-8

# In[11]:


import gc
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import skimage
from skimage import data, io, filters, morphology, feature, util, transform
#from skimage.feature import register_translation
#from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

#user var
#basePath = "/home/francesco/Downloads/imgs_orig32bit_orig8bit_outFeat/output_feat_linee/"
basePath = "/home/francesco/Seafile/PHD/imageAlign/testPy/output_feat_lines/"
imgNum1 = 5 # riferimento per Y
imgNum2 = 6 # Immagine da registrare su una pre-esistente 
#imgNum3 = 6 # riferimento X

ImgPath1 = "featExt" + str(imgNum1) + ".png"
ImgPath2 = "featExt" + str(imgNum2) + ".png"
#ImgPath3 = "featExt" + str(imgNum3) + ".png"
print("Loading images")
image1 = io.imread(basePath+ImgPath1)
image2 = io.imread(basePath+ImgPath2)
#image3 = io.imread(basePath+ImgPath3)
#fig setup
fig = plt.figure(figsize=(12, 9))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1, adjustable='box-forced')

ax1.imshow(image1, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference Image:'+str(imgNum1))

ax2.imshow(image2, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Image to align:' + str(imgNum2))

"""
ax3.imshow(image3, cmap='gray')
ax3.set_axis_off()
ax3.set_title('Image3 X')
"""



print("Detecting...")
shiftY, errorY, diffphaseY = skimage.feature.register_translation(image1, image2, 1)
#shiftX, errorX, diffphaseX = skimage.feature.register_translation(image3, image2, 1000)
print("Detected subpixel offset (y, x): {}".format(shiftY))
print("Shifting...")

offset_imageY = fourier_shift(np.fft.fftn(image2), shiftY)
offset_imageY = np.fft.ifftn(offset_imageY)
offset_imageY = offset_imageY.real
#image1 = image1.astype(np.float)
offset_imageY = (offset_imageY-np.amin(offset_imageY))/np.amax(offset_imageY)

"""
offset_imageX = fourier_shift(np.fft.fftn(image3), shiftX)
offset_imageX = np.fft.ifftn(offset_imageX)
offset_imageX = offset_imageX.real
"""

'''
tform = skimage.transform.SimilarityTransform(translation = [shift[1],shift[0]])
offset_image1 = skimage.transform.warp(image2,tform)
'''

image1 = image1.astype(np.float)
image1 = (image1-np.amin(image1))/np.amax(image1)

#offset_imageX = offset_imageX.astype(np.float)
#offset_imageX = offset_imageX/np.amax(offset_imageX)

ax3.imshow(offset_imageY, cmap='gray')
ax3.set_axis_off()
ax3.set_title('Image translated')

plt.show()

"""
offset_imageY = offset_imageY.astype(np.float)
offset_imageY = offset_imageY/np.amax(offset_imageY)
"""
#io.imsave('tmpOffset.png',offset_image1)
#offset_image = io.imread('tmpOffset.png')
#offset_image = offset_image1.astype(np.float)/np.amax(offset_image)
'''
offset_image_color = skimage.color.gray2rgb(offset_image1) * [1,0,0]
offset_image_color = offset_image1_color.astype(np.float)
image1color = skimage.color.gray2rgb(image1) * [0,0,1]
image1color = image1Color.astype(np.float)
'''

#Overlapped = (offset_imageY + image1 + offset_imageX)*0.0 + (offset_imageY * image1 * offset_imageX)*0.99
ZeroTensor = np.zeros((image1.shape[0],image1.shape[1],3)) #color image
redTensor = ZeroTensor.copy()
blueTensor = ZeroTensor.copy()

redTensor[:,:,0] = ((offset_imageY/np.amax(offset_imageY)) >0.1).astype(np.float)
blueTensor[:,:,2] = ((image1/np.amax(image1)) >0.1).astype(np.float)

Overlapped = redTensor + blueTensor
#Overlapped = (offset_imageY + image1)*0.2 + (offset_imageY * image1)*0.8
#Overlapped = Overlapped/np.amax(Overlapped)
#Overlapped = Overlapped.astype(np.float)

fig2 = plt.figure(figsize=(24, 18))
ax4 = plt.subplot(1, 3, 3,sharex=ax1, sharey=ax1, adjustable='box-forced')
#ax4.imshow(Overlapped, cmap = 'gray')
ax4.imshow(Overlapped)
#ax3.imshow(image1color)
ax4.set_axis_off()
ax4.set_title('Ref. Image x Shifted image2')
plt.show()
io.imsave("overlapped_lines/im_" + str(imgNum1) + "_" + str(imgNum2) + ".png",Overlapped)

transFileId = open("overlapped_lines/pos_" + str(imgNum1) + "_" + str(imgNum2) + ".txt","w") 
transFileId.write("[{x};{y}]".format(x=shiftY[1],y=shiftY[0])) 
transFileId.close()

gc.collect()

print('done')

#xraw = [0, 229.0, 229.0, 27.0, 0, 228.0, 229.0, 227.0, 0, 228.0, 229.0, 228.0]
#yraw = [0, 0.0, -1.0, -186.0, 0, -2.0, -3.0, 0.0, 0, 0.0, 1.0, 1.0]


