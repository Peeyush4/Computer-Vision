import numpy as np
import scipy as sp
import cv2

def computeQuantizationError(origImg, quantizedImg):
    return np.sum(np.square(origImg - quantizedImg))