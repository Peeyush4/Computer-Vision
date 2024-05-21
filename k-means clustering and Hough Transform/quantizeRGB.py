import numpy as np
import scipy as sp
import cv2

def quantizeRGB(origImg, k):
    img = origImg.reshape((-1, 3))

    #k-means
    inititalCenters = cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(img.astype(np.float32), 
                                              K=k,
                                              bestLabels=None,
                                              criteria=criteria, 
                                              attempts=10, 
                                              flags=inititalCenters) 
    
    #After finding kmeans, creating image
    centers = centers.astype(np.uint8)
    res = centers[labels.flatten()]
    result = res.reshape((origImg.shape))
    
    return result, centers
