import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

imgSrc = 'images\\flowers.jpg'
gray_img = cv.imread(imgSrc,cv.IMREAD_GRAYSCALE)
h,w = gray_img.shape    # dim imaginii h - height, w - width

blur = cv.bilateralFilter(gray_img,3,50,50)

# MATH
# SOBEL
# filters --- kernel = 3
horizontal_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
vertical_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  

sobelImage = np.zeros((h, w))
# SCHARR
# filters --- kernel = 3
horizontal_scharr = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
vertical_scharr = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) 

scharrImage = np.zeros((h, w))

# PREWITT
# filters --- kernel = 3
horizontal_prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  
vertical_prewitt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) 

prewittImage = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGradSobel = (horizontal_sobel[0, 0] * blur[i - 1, j - 1]) + \
                         (horizontal_sobel[0, 1] * blur[i - 1, j]) + \
                         (horizontal_sobel[0, 2] * blur[i - 1, j + 1]) + \
                         (horizontal_sobel[1, 0] * blur[i, j - 1]) + \
                         (horizontal_sobel[1, 1] * blur[i, j]) + \
                         (horizontal_sobel[1, 2] * blur[i, j + 1]) + \
                         (horizontal_sobel[2, 0] * blur[i + 1, j - 1]) + \
                         (horizontal_sobel[2, 1] * blur[i + 1, j]) + \
                         (horizontal_sobel[2, 2] * blur[i + 1, j + 1])

        verticalGradSobel = (vertical_sobel[0, 0] * blur[i - 1, j - 1]) + \
                       (vertical_sobel[0, 1] * blur[i - 1, j]) + \
                       (vertical_sobel[0, 2] * blur[i - 1, j + 1]) + \
                       (vertical_sobel[1, 0] * blur[i, j - 1]) + \
                       (vertical_sobel[1, 1] * blur[i, j]) + \
                       (vertical_sobel[1, 2] * blur[i, j + 1]) + \
                       (vertical_sobel[2, 0] * blur[i + 1, j - 1]) + \
                       (vertical_sobel[2, 1] * blur[i + 1, j]) + \
                       (vertical_sobel[2, 2] * blur[i + 1, j + 1])
        # Sobel Edge Magnitude
        sobelImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradSobel, 2.0) + pow(verticalGradSobel, 2.0))


        horizontalGradScharr = (horizontal_scharr[0, 0] * blur[i - 1, j - 1]) + \
                         (horizontal_scharr[0, 1] * blur[i - 1, j]) + \
                         (horizontal_scharr[0, 2] * blur[i - 1, j + 1]) + \
                         (horizontal_scharr[1, 0] * blur[i, j - 1]) + \
                         (horizontal_scharr[1, 1] * blur[i, j]) + \
                         (horizontal_scharr[1, 2] * blur[i, j + 1]) + \
                         (horizontal_scharr[2, 0] * blur[i + 1, j - 1]) + \
                         (horizontal_scharr[2, 1] * blur[i + 1, j]) + \
                         (horizontal_scharr[2, 2] * blur[i + 1, j + 1])

        verticalGradScharr = (vertical_scharr[0, 0] * blur[i - 1, j - 1]) + \
                       (vertical_scharr[0, 1] * blur[i - 1, j]) + \
                       (vertical_scharr[0, 2] * blur[i - 1, j + 1]) + \
                       (vertical_scharr[1, 0] * blur[i, j - 1]) + \
                       (vertical_scharr[1, 1] * blur[i, j]) + \
                       (vertical_scharr[1, 2] * blur[i, j + 1]) + \
                       (vertical_scharr[2, 0] * blur[i + 1, j - 1]) + \
                       (vertical_scharr[2, 1] * blur[i + 1, j]) + \
                       (vertical_scharr[2, 2] * blur[i + 1, j + 1])
        # Scharr Edge Magnitude
        scharrImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradScharr, 2.0) + pow(verticalGradScharr, 2.0))


        horizontalGradPrewitt = (horizontal_prewitt[0, 0] * blur[i - 1, j - 1]) + \
                         (horizontal_prewitt[0, 1] * blur[i - 1, j]) + \
                         (horizontal_prewitt[0, 2] * blur[i - 1, j + 1]) + \
                         (horizontal_prewitt[1, 0] * blur[i, j - 1]) + \
                         (horizontal_prewitt[1, 1] * blur[i, j]) + \
                         (horizontal_prewitt[1, 2] * blur[i, j + 1]) + \
                         (horizontal_prewitt[2, 0] * blur[i + 1, j - 1]) + \
                         (horizontal_prewitt[2, 1] * blur[i + 1, j]) + \
                         (horizontal_prewitt[2, 2] * blur[i + 1, j + 1])

        verticalGradPrewitt = (vertical_prewitt[0, 0] * blur[i - 1, j - 1]) + \
                       (vertical_prewitt[0, 1] * blur[i - 1, j]) + \
                       (vertical_prewitt[0, 2] * blur[i - 1, j + 1]) + \
                       (vertical_prewitt[1, 0] * blur[i, j - 1]) + \
                       (vertical_prewitt[1, 1] * blur[i, j]) + \
                       (vertical_prewitt[1, 2] * blur[i, j + 1]) + \
                       (vertical_prewitt[2, 0] * blur[i + 1, j - 1]) + \
                       (vertical_prewitt[2, 1] * blur[i + 1, j]) + \
                       (vertical_prewitt[2, 2] * blur[i + 1, j + 1])
        # Prewitt Edge Magnitude
        prewittImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradPrewitt, 2.0) + pow(verticalGradPrewitt, 2.0))


plt.imsave('res\sobelResMath.png', sobelImage, cmap='gray', format='png')
plt.imsave('res\scharrResMath.png', scharrImage, cmap='gray', format='png')
plt.imsave('res\prewittResMath.png', prewittImage, cmap='gray', format='png')

gray = cv.imread('res\sobelResMath.png',cv.IMREAD_GRAYSCALE)
ret1,th1 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
gray = cv.imread('res\scharrResMath.png',cv.IMREAD_GRAYSCALE)
ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
gray = cv.imread('res\prewittResMath.png',cv.IMREAD_GRAYSCALE)
ret3,th3 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imsave('res\TsobelResMath.png', th1, cmap='gray', format='png')
plt.imsave('res\TscharrResMath.png', th2, cmap='gray', format='png')
plt.imsave('res\TprewittResMath.png', th3, cmap='gray', format='png')
os.remove('res\sobelResMath.png')
os.remove('res\scharrResMath.png')
os.remove('res\prewittResMath.png')

# ------------ OPENCV
# SOBEL
#denoise img
sobel_x3 = cv.Sobel(blur,cv.CV_8U,1,0,ksize=3)
sobel_y3 = cv.Sobel(blur,cv.CV_8U,0,1,ksize=3)
sobel_dst3 = cv.addWeighted(sobel_x3,0.5,sobel_y3,0.5,0)

sobel_x5 = cv.Sobel(blur,cv.CV_8U,1,0,ksize=5)
sobel_y5 = cv.Sobel(blur,cv.CV_8U,0,1,ksize=5)
sobel_dst5 = cv.addWeighted(sobel_x5,0.5,sobel_y5,0.5,0)

ret4,th4 = cv.threshold(sobel_dst3,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret5,th5 = cv.threshold(sobel_dst5,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#plt.imsave('res\sobelResOcv3.png', sobel_dst3, cmap='gray', format='png')
#plt.imsave('res\sobelResOcv5.png', sobel_dst5, cmap='gray', format='png')
plt.imsave('res\TsobelResOcv3.png', th4, cmap='gray', format='png')
plt.imsave('res\TsobelResOcv5.png', th5, cmap='gray', format='png')

# SCHARR
scharr_x = cv.Scharr(blur,cv.CV_8U,1,0)
scharr_y = cv.Scharr(blur,cv.CV_8U,0,1)
scharr_dst = cv.addWeighted(scharr_x,0.5,scharr_y,0.5,0)
ret6,th6 = cv.threshold(scharr_dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#plt.imsave('res\scharrResOcv.png', scharr_dst, cmap='gray', format='png')
plt.imsave('res\TscharrResOcv.png', th6, cmap='gray', format='png')

# PREWITT
prewittx = cv.filter2D(blur, -1, horizontal_prewitt)
prewitty = cv.filter2D(blur, -1, vertical_prewitt)
prewitt_dst = cv.addWeighted(prewittx,0.5,prewitty,0.5,0)
ret7,th7 = cv.threshold(prewitt_dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#plt.imsave('res\prewittResOcv.png', prewitt_dst, cmap='gray', format='png')
plt.imsave('res\TprewittResOcv.png', th7, cmap='gray', format='png')

# LAPLACIAN
laplacianImage = cv.Laplacian(blur, cv.CV_8U, None, 3);
ret8,th8 = cv.threshold(laplacianImage,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#plt.imsave('res\prewittResOcv.png', laplacianImage, cmap='gray', format='png')
plt.imsave('res\TprewittResOcv.png', th8, cmap='gray', format='png')

# CANNY
cannyImage = cv.Canny(gray_img,50,150,L2gradient=True)
plt.imsave('res\TcannyResOcv.png', cannyImage, cmap='gray', format='png')