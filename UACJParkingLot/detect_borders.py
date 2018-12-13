import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_12-35-19.jpg')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

kernel = np.array([[8,-1,-1],[-1,8,-1],[-1,-1,8]])
# we use 'valid' which means we do not add zero padding to our image
edges = scipy.signal.convolve2d(img, kernel, 'valid')
#print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(img/np.max(np.abs(edges)), clip_limit=0.03)

plt.subplot(121),plt.imshow(img, cmap=plt.cm.gray),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges_equalized, cmap=plt.cm.gray),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()

plt.imsave('C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\test.jpg', edges_equalized, cmap=plt.cm.gray)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_12-35-19.jpg')
#
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
#
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()