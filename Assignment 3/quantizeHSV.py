import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt

def quantizeHSV(origImg, k):
    hueImage = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
    #Kmeans
    inititalCenters = cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(hueImage[:,:,0].reshape(-1).astype(np.float32), 
                                              K=k,
                                              bestLabels=None,
                                              criteria=criteria, 
                                              attempts=10, 
                                              flags=inititalCenters) 
    
    #After finding kmeans, creating image
    centers = centers.astype(np.uint8)
    res = centers[labels.flatten()]
    
    #Replacing with common hue
    hueImage[:, :, 0] = res.reshape((hueImage.shape[0], hueImage.shape[1]))
    hueImage = cv2.cvtColor(hueImage, cv2.COLOR_HSV2BGR)
    return hueImage, centers

