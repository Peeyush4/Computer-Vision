from computeH import computeH
import numpy as np
import matplotlib.pyplot as plt
import cv2
from readImage import readImage
from warpImage import warpImage
#Field
inputIm, refIm = cv2.imread('crop1.jpg'), cv2.imread('crop2.jpg')
points_1, points_2 = np.load('cc1.npy').T, np.load('cc2.npy').T

H = computeH(points_1, points_2)
warpIm, mergeIm = warpImage(inputIm=inputIm, refIm=refIm, H=H)
cv2.imshow('Warped Image', warpIm)
cv2.waitKey(0)

cv2.imshow('Merged Image', mergeIm)
cv2.waitKey(0)

#WDC
inputIm, refIm = cv2.imread('wdc1.jpg'), cv2.imread('wdc2.jpg')
points_1, points_2 = np.load('points.npy')

H = computeH(points_1, points_2)
warpIm, mergeIm = warpImage(inputIm=inputIm, refIm=refIm, H=H)
cv2.imshow('Warped Image', warpIm)
cv2.waitKey(0)

cv2.imshow('Merged Image', mergeIm)
cv2.waitKey(0)
cv2.imwrite('wdc_merged.png', mergeIm)

#Nature
inputIm, refIm = cv2.imread('nature1.jpg'), cv2.imread('nature2.jpg')
points_1, points_2 = readImage(inputIm, refIm)
# points_1, points_2 = np.load('nature.npy')

H = computeH(points_1, points_2)
warpIm, mergeIm = warpImage(inputIm=inputIm, refIm=refIm, H=H)
cv2.imshow('Warped Image', warpIm)
cv2.waitKey(0)

cv2.imshow('Merged Image', mergeIm)
cv2.waitKey(0)
cv2.imwrite('nature_merged.png', mergeIm)