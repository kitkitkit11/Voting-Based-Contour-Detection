import cv2
import numpy as np
from matplotlib import pyplot as plt

imgSrc = 'images\car.jpg'

# OpenCV
img = cv2.imread(imgSrc,cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape

#denoise img
denoised = cv2.fastNlMeansDenoising(img,None,3,7,21)

#sobel x axis
sobelx0 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=-1)
sobelx1 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=1)
sobelx3 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=3)
sobelx5 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=5)
sobelx7 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=7)

#sobelx0 = np.uint8(np.absolute(sobel64x0))
#sobelx1 = np.uint8(np.absolute(sobel64x1))
#sobelx3 = np.uint8(np.absolute(sobel64x3))
#sobelx5 = np.uint8(np.absolute(sobel64x5))
#sobelx7 = np.uint8(np.absolute(sobel64x7))

#sobel y axis
sobely0 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=-1)    # CV_C64
sobely1 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=1)
sobely3 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=3)
sobely5 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=5)
sobely7 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=7)

#sobely0 = np.uint8(np.absolute(sobel64y0))
#sobely1 = np.uint8(np.absolute(sobel64y1))
#sobely3 = np.uint8(np.absolute(sobel64y3))
#sobely5 = np.uint8(np.absolute(sobel64y5))
#sobely7 = np.uint8(np.absolute(sobel64y7))

# sobel both axes
dst0 = cv2.addWeighted(sobelx0,0.5,sobely0,0.5,0)
dst1 = cv2.addWeighted(sobelx1,0.5,sobely1,0.5,0)
dst3 = cv2.addWeighted(sobelx3,0.5,sobely3,0.5,0)
dst5 = cv2.addWeighted(sobelx5,0.5,sobely5,0.5,0)
dst7 = cv2.addWeighted(sobelx7,0.5,sobely7,0.5,0)

# Otsu's thresholding after Gaussian filtering
blur0 = cv2.GaussianBlur(dst0,(3,3),0)
blur1 = cv2.GaussianBlur(dst1,(3,3),0)
blur3 = cv2.GaussianBlur(dst3,(3,3),0)
blur5 = cv2.GaussianBlur(dst5,(3,3),0)
blur7 = cv2.GaussianBlur(dst7,(3,3),0)

ret0,th0 = cv2.threshold(blur0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret1,th1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret5,th5 = cv2.threshold(blur5,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret7,th7 = cv2.threshold(blur7,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#rez
plt.subplot(2,3,1),plt.imshow(denoised,cmap = 'gray')
plt.title('OriginalD'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx0,cmap = 'gray')
plt.title('SharrX'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely0,cmap = 'gray')
plt.title('SharrY'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(dst0,cmap = 'gray')
plt.title('SharrXY'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur0,cmap = 'gray')
plt.title('SharrB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th0,cmap = 'gray')
plt.title('SharrT'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(2,3,1),plt.imshow(denoised,cmap = 'gray')
plt.title('OriginalD'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx1,cmap = 'gray')
plt.title('SobelX1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely1,cmap = 'gray')
plt.title('SobelY1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(dst1,cmap = 'gray')
plt.title('SobelXY1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur1,cmap = 'gray')
plt.title('SobelB1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th1,cmap = 'gray')
plt.title('SobelT1'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(2,3,1),plt.imshow(denoised,cmap = 'gray')
plt.title('OriginalD'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx3,cmap = 'gray')
plt.title('SobelX3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely3,cmap = 'gray')
plt.title('SobelY3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(dst3,cmap = 'gray')
plt.title('SobelXY3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur3,cmap = 'gray')
plt.title('SobelB3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th3,cmap = 'gray')
plt.title('SobelT3'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(2,3,1),plt.imshow(denoised,cmap = 'gray')
plt.title('OriginalD'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx5,cmap = 'gray')
plt.title('SobelX5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely5,cmap = 'gray')
plt.title('SobelY5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(dst5,cmap = 'gray')
plt.title('SobelXY5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur5,cmap = 'gray')
plt.title('SobelB5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th5,cmap = 'gray')
plt.title('SobelT5'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(2,3,1),plt.imshow(denoised,cmap = 'gray')
plt.title('OriginalD'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx7,cmap = 'gray')
plt.title('SobelX7'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely7,cmap = 'gray')
plt.title('SobelY7'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(dst7,cmap = 'gray')
plt.title('SobelXY7'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur7,cmap = 'gray')
plt.title('SobelB7'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(th7,cmap = 'gray')
plt.title('SobelT7'), plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis

plt.show()
