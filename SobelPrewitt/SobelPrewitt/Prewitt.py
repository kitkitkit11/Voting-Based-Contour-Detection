import cv2
import numpy as np
from matplotlib import pyplot as plt

imgSrc = 'images\nature.jpg'
# OpenCV
img = cv2.imread(imgSrc,cv2.IMREAD_GRAYSCALE)

img_gaussian = cv2.GaussianBlur(img,(3,3),0)

#prewitt3
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
prewitty = cv2.filter2D(img_gaussian, -1, kernely)
prewittxy = prewittx + prewitty


#rez
plt.subplot(1,4,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2),plt.imshow(prewittx,cmap = 'gray')
plt.title('PrewittX'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3),plt.imshow(prewitty,cmap = 'gray')
plt.title('PrewittY'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4),plt.imshow(prewittxy,cmap = 'gray')
plt.title('PrewittXY'), plt.xticks([]), plt.yticks([])

plt.show()
