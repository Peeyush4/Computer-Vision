from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from find_optimal_vertical_seam import find_optimal_vertical_seam
import copy
import matplotlib.pyplot as plt
import cv2

def displaySeam(im, seam, type='HORIZONTAL'):
    image = copy.copy(im)  
    image[seam[:, 0], seam[:, 1]] = [0, 0, 255]
    
    plt.imshow(image)
    plt.show()
    return image
