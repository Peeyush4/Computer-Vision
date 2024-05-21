import scipy as sp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from find_optimal_horizontal_seam import find_optimal_horizontal_seam

def reduceHeight(im, energyImage):
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, type)
    horizontalSeam = find_optimal_horizontal_seam(cumulativeEnergyMap)
    
    #Image size reduction
    requiredPixels = np.ones(im.shape).astype(bool)
    requiredPixels[horizontalSeam[:, 0], horizontalSeam[:, 1]] = np.array([False] * 3)
    reducedImage = im[requiredPixels].reshape(im.shape[0] - 1, im.shape[1], 3)

    #Energy reduction
    requiredEnergy = np.ones(energyImage.shape).astype(bool)
    requiredEnergy[horizontalSeam[:, 0], horizontalSeam[:, 1]] = False
    reducedEnergy = energyImage[requiredEnergy].reshape(im.shape[0] - 1, im.shape[1])
    return reducedImage, reducedEnergy