from quantizeHSV import quantizeHSV
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

def getHueHists(im, k):
    hueImage = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hueData = hueImage[:, :, 0].reshape(-1)

    histEqual = plt.figure()
    plt.hist(hueData, bins=range(0, 180, 180//(k + 1)))    
    plt.title(f"Hue Equal Bin Histogram for k={k}")
    plt.title(f"Hue Data Histogram")
    plt.xlabel('Hue values')
    plt.ylabel('Frequency')

    hueClusteredRGBImage, hueClusteredCenters = quantizeHSV(im, k)
    hueClusterInfo = cv2.cvtColor(hueClusteredRGBImage, cv2.COLOR_BGR2HSV)[:, :, 0].reshape(-1)

    histClustered = plt.figure()
    plt.hist(hueClusterInfo, bins=np.append(np.sort(hueClusteredCenters.reshape(-1)), 180))
    plt.title(f"k-means hue histogram for k={k}")
    plt.xlabel('Clustered hue values')
    plt.ylabel('Frequency')
    
    return histEqual, histClustered

