
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import data, io, filters, morphology, feature, util, transform
#from skimage.feature import register_translation
#from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

#user var
#basePath = "/home/francesco/Downloads/imgs_orig32bit_orig8bit_outFeat/output_feat_linee/"
basePath = "/home/francesco/Seafile/PHD/imageAlign/testPy/output_clahe/"
imgNum1 = 3
imgNum2 = 4

ImgPath1 = "Output_"+str(imgNum1)+".png"
ImgPath2 = "Output_"+str(imgNum2)+".png"
print("Loading images")
image1 = io.imread(basePath+ImgPath1)
image2 = io.imread(basePath+ImgPath2)

#fig setup
fig = plt.figure(figsize=(12, 9))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')

ax1.imshow(image1, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Image1')

ax2.imshow(image2, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Image2')

plt.show()

print("Detecting...")
shift, error, diffphase = skimage.feature.register_translation(image1, image2, 1000)
print("Detected subpixel offset (y, x): {}".format(shift))
print("Shifting...")

offset_image = fourier_shift(np.fft.fftn(image2), shift)
offset_image = np.fft.ifftn(offset_image)
offset_image = offset_image.real

##my shift
corrMask = corr2dFreq(img1>0,img2>0)
#corrMask = scipy.signal.correlate2d((img1.data>0).astype(np.uint32), (img2.data>0).astype(np.uint32), mode='same', boundary='fill', fillvalue=0)
outImg = corrImg/((corrMask**2)+np.sum(np.ravel(img2)**2))
#outImg[outImg < 0] = 0
#outImg **= 2
#tf.imsave("corrNorm.tiff", outImg.astype(np.float32))
#outImg /= np.amax(outImg)
print('displaying...')
#outImg = outImg/np.amax(outImg)
print('outimage shape', outImg.shape)
#indexPos = np.unravel_index(np.argmax(outImg), outImg.shape)
local_maxi = skimage.feature.peak_local_max(outImg,
                                            min_distance = int(round(SCALESIZE/30)),
                                            threshold_rel=np.mean(outImg.ravel()) -
                                                        np.std(outImg.ravel())/4)


xMaxPos = local_maxi[:,1]
yMaxPos = local_maxi[:,0]

fig2 = plt.pyplot.figure(figsize=(14, 6))  # ,dpi=100)
ax3 = plt.pyplot.subplot(1, 2, 1, adjustable='box-forced')
ax3.imshow(outImg,cmap='gray')
ax3.set_title("Norm Corr")
ax3.plot(xMaxPos, yMaxPos, 'ro')
#ax3.text(xMaxPos, yMaxPos, str(local_maxi), fontsize=15, color='red')
print('From normalized corr\n',local_maxi)

###


'''
tform = skimage.transform.SimilarityTransform(translation = [shift[1],shift[0]])
offset_image1 = skimage.transform.warp(image2,tform)
'''

image1 = image1.astype(np.float)
image1 = image1/np.amax(image1)
image1 = 1 - image1

offset_image = offset_image.astype(np.float)
offset_image = offset_image/np.amax(offset_image)
offset_image = 1 - offset_image

#io.imsave('tmpOffset.png',offset_image1)
#offset_image = io.imread('tmpOffset.png')
#offset_image = offset_image1.astype(np.float)/np.amax(offset_image)
'''
offset_image_color = skimage.color.gray2rgb(offset_image1) * [1,0,0]
offset_image_color = offset_image1_color.astype(np.float)
image1color = skimage.color.gray2rgb(image1) * [0,0,1]
image1color = image1Color.astype(np.float)
'''

Overlapped = (offset_image + image1)*0.2 + (offset_image * image1)*0.8
Overlapped = Overlapped/np.amax(Overlapped)
#Overlapped = Overlapped.astype(np.float)

fig2 = plt.figure(figsize=(24, 18))
ax3 = plt.subplot(1, 3, 3)
ax3.imshow(Overlapped, cmap = 'gray')
#ax3.imshow(image1color)
ax3.set_axis_off()
ax3.set_title('Image1 x Shifted image2')
plt.show()
io.imsave("overlapped_dots/im_" + str(imgNum1) + "_" + str(imgNum2) + ".png",Overlapped)

transFileId = open("overlapped_dots/pos_" + str(imgNum1) + "_" + str(imgNum2) + ".txt","w") 
transFileId.write("{}".format(shift)) 
transFileId.close()

print('done')

