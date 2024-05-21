import scipy as sp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from find_optimal_vertical_seam import find_optimal_vertical_seam

def reduceWidth(im, energyImage):
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, type)
    verticalSeam = find_optimal_vertical_seam(cumulativeEnergyMap)
    
    #Image size reduction
    requiredPixels = np.ones(im.shape).astype(bool)
    requiredPixels[verticalSeam[:, 0], verticalSeam[:, 1]] = np.array([False] * 3)
    reducedImage = im[requiredPixels].reshape(im.shape[0], im.shape[1] - 1, 3)

    #Energy reduction
    requiredEnergy = np.ones(energyImage.shape).astype(bool)
    requiredEnergy[verticalSeam[:, 0], verticalSeam[:, 1]] = False
    reducedEnergy = energyImage[requiredEnergy].reshape(im.shape[0], im.shape[1] - 1)
    return reducedImage, reducedEnergy