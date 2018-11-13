import cv2
import numpy as np
from matplotlib import pyplot as plt

imgSrc = 'images\car.jpg'
gray_img = cv2.imread(imgSrc,cv2.IMREAD_GRAYSCALE)
h,w = gray_img.shape    # dim imaginii h - height, w - width

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
        horizontalGradSobel = (horizontal_sobel[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal_sobel[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal_sobel[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal_sobel[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal_sobel[1, 1] * gray_img[i, j]) + \
                         (horizontal_sobel[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal_sobel[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal_sobel[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal_sobel[2, 2] * gray_img[i + 1, j + 1])

        verticalGradSobel = (vertical_sobel[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical_sobel[0, 1] * gray_img[i - 1, j]) + \
                       (vertical_sobel[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical_sobel[1, 0] * gray_img[i, j - 1]) + \
                       (vertical_sobel[1, 1] * gray_img[i, j]) + \
                       (vertical_sobel[1, 2] * gray_img[i, j + 1]) + \
                       (vertical_sobel[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical_sobel[2, 1] * gray_img[i + 1, j]) + \
                       (vertical_sobel[2, 2] * gray_img[i + 1, j + 1])
        # Sobel Edge Magnitude
        sobelImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradSobel, 2.0) + pow(verticalGradSobel, 2.0))


        horizontalGradScharr = (horizontal_scharr[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal_scharr[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal_scharr[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal_scharr[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal_scharr[1, 1] * gray_img[i, j]) + \
                         (horizontal_scharr[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal_scharr[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal_scharr[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal_scharr[2, 2] * gray_img[i + 1, j + 1])

        verticalGradScharr = (vertical_scharr[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical_scharr[0, 1] * gray_img[i - 1, j]) + \
                       (vertical_scharr[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical_scharr[1, 0] * gray_img[i, j - 1]) + \
                       (vertical_scharr[1, 1] * gray_img[i, j]) + \
                       (vertical_scharr[1, 2] * gray_img[i, j + 1]) + \
                       (vertical_scharr[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical_scharr[2, 1] * gray_img[i + 1, j]) + \
                       (vertical_scharr[2, 2] * gray_img[i + 1, j + 1])
        # Prewitt Edge Magnitude
        scharrImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradScharr, 2.0) + pow(verticalGradScharr, 2.0))


        horizontalGradPrewitt = (horizontal_prewitt[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal_prewitt[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal_prewitt[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal_prewitt[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal_prewitt[1, 1] * gray_img[i, j]) + \
                         (horizontal_prewitt[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal_prewitt[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal_prewitt[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal_prewitt[2, 2] * gray_img[i + 1, j + 1])

        verticalGradPrewitt = (vertical_prewitt[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical_prewitt[0, 1] * gray_img[i - 1, j]) + \
                       (vertical_prewitt[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical_prewitt[1, 0] * gray_img[i, j - 1]) + \
                       (vertical_prewitt[1, 1] * gray_img[i, j]) + \
                       (vertical_prewitt[1, 2] * gray_img[i, j + 1]) + \
                       (vertical_prewitt[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical_prewitt[2, 1] * gray_img[i + 1, j]) + \
                       (vertical_prewitt[2, 2] * gray_img[i + 1, j + 1])
        # Prewitt Edge Magnitude
        prewittImage[i - 1, j - 1] = np.sqrt(pow(horizontalGradPrewitt, 2.0) + pow(verticalGradPrewitt, 2.0))

 

plt.imsave('res\sobelResMath.png', sobelImage, cmap='gray', format='png')
plt.imsave('res\scharrResMath.png', scharrImage, cmap='gray', format='png')
plt.imsave('res\prewittResMath.png', prewittImage, cmap='gray', format='png')

# ------------ OPENCV

# SOBEL
#denoise img
denoised = cv2.fastNlMeansDenoising(gray_img,None,3,7,21)

sobel_x3 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=3)
sobel_y3 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=3)
sobel_dst3 = cv2.addWeighted(sobel_x3,0.5,sobel_y3,0.5,0)

sobel_x5 = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=5)
sobel_y5 = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=5)
sobel_dst5 = cv2.addWeighted(sobel_x5,0.5,sobel_y5,0.5,0)

plt.imsave('res\sobelResOcv3.png', sobel_dst3, cmap='gray', format='png')
plt.imsave('res\sobelResOcv5.png', sobel_dst5, cmap='gray', format='png')

# SCHARR
scharr_x = cv2.Sobel(denoised,cv2.CV_8U,1,0,ksize=-1)
scharr_y = cv2.Sobel(denoised,cv2.CV_8U,0,1,ksize=-1)
scharr_dst = cv2.addWeighted(scharr_x,0.5,scharr_y,0.5,0)

plt.imsave('res\scharrResOcv.png', sobel_dst5, cmap='gray', format='png')

# PREWITT
blur_gaussian = cv2.GaussianBlur(gray_img, (3, 3), 0)

prewittx = cv2.filter2D(blur_gaussian, -1, horizontal_prewitt)
prewitty = cv2.filter2D(blur_gaussian, -1, vertical_prewitt)
prewitt_dst = prewittx + prewitty

plt.imsave('res\prewittResOcv.png', prewitt_dst, cmap='gray', format='png')

# LAPLACIAN
laplacianImage = cv2.Laplacian(blur_gaussian, cv2.CV_8U)

# CANNY
cannyImage1 = cv2.Canny(gray_img,100,200,L2gradient=True)
plt.imsave('res\canny100-200.png', cannyImage1, cmap='gray', format='png')

cannyImage2 = cv2.Canny(gray_img,100,300,L2gradient=True)
plt.imsave('res\canny100-300.png', cannyImage2, cmap='gray', format='png')

cannyImage3 = cv2.Canny(gray_img,200,300,L2gradient=True)
plt.imsave('res\canny200-300.png', cannyImage3, cmap='gray', format='png')

plt.subplot(2,3,1),plt.imshow(sobelImage,cmap = 'gray')
plt.title('SobelMath'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(scharrImage,cmap = 'gray')
plt.title('ScharrMath'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(prewittImage,cmap = 'gray')
plt.title('PrewittMath'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobel_dst3,cmap = 'gray')
plt.title('SobelOpenCV3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(scharr_dst,cmap = 'gray')
plt.title('ScharrOpenCV'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(prewitt_dst,cmap = 'gray')
plt.title('PrewittOpenCV'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(2,2,1),plt.imshow(sobel_dst5,cmap = 'gray')
plt.title('SobelOpenCV5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(cannyImage1,cmap = 'gray')
plt.title('Canny100-200'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(cannyImage2,cmap = 'gray')
plt.title('Canny100-300'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(cannyImage3,cmap = 'gray')
plt.title('Canny200-300'), plt.xticks([]), plt.yticks([])

plt.show()