from glob import glob
import os
from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)


def load_images_from_folder(folder):
    imagesArr = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename))

        if img is not None:
            imagesArr.append(img)

    return imagesArr


images = load_images_from_folder('./res')

# weighted voting
quota = 45

voters = len(images)
weightOfVoters = [10, 30, 10, 5, 10, 7, 5, 5, 5]
quota = 50

# threshold images from grayscale to binary B/W
thresholdLevel = 70
thresholdImages = []
i = 0
for image in images:
    (t, threshold_img) = cv2.threshold(
        image, thresholdLevel, 255, cv2.THRESH_BINARY)
    thresholdImages.append(threshold_img)
    i = i+1
    cv2.imwrite("test" + str(i) + ".png", threshold_img)

# img = cv2.imread(os.listdir('./res/')[0], 0)
img = cv2.imread('./res/canny100-200.png', 0)
rows, cols = img.shape

majVoteResult = np.zeros((rows, cols, 3))
unanimousVoteResult = np.zeros((rows, cols, 3))
weightedVoteResult = np.zeros((rows, cols, 3))

# compose list of votes per pixel from each resulting image from each algorithm
for i in range(rows):
    for j in range(cols):
        votesPerPixel = []
        for image in thresholdImages:
            pixel = image[i, j]

            if (pixel == [255, 255, 255]).all():
                votesPerPixel.append(1)
            else:
                votesPerPixel.append(0)

        # majority voting
        c = Counter(votesPerPixel)
        value, count = c.most_common()[0]

        if value == 1:
            majVoteResult[i, j] = np.array([255, 255, 255])
        else:
            majVoteResult[i, j] = np.array([0, 0, 0])

        # unanimous voting
        if count == len(votesPerPixel):
            if value == 1:
                unanimousVoteResult[i, j] = np.array([255, 255, 255])
            else:
                unanimousVoteResult[i, j] = np.array([0, 0, 0])
        else:
            unanimousVoteResult[i, j] = np.array([0, 0, 0])

        # weighted voting
        sum = 0
        for idx, val in enumerate(votesPerPixel):
            sum += val * weightOfVoters[idx]
            if sum >= quota:
                weightedVoteResult[i, j] = np.array([255, 255, 255])
            else:
                weightedVoteResult[i, j] = np.array([0, 0, 0])

cv2.imwrite('MajVoteResult.png', majVoteResult)
cv2.imwrite('UanimousVoteResult.png', unanimousVoteResult)
cv2.imwrite('WeightedVoteResult.png', weightedVoteResult)
