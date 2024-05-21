import numpy as np
from find_optimal_vertical_seam import find_optimal_vertical_seam
import copy

def find_optimal_horizontal_seam(cumulativeEnergyMap):
    #Input cumulativeEnergyMap: np array of size MxN
    #Output: Pixels of size Nx2
    # M = np.rot90(cumulativeEnergyMap)
    # pixels = find_optimal_vertical_seam(M)
    # pixels = pixels[:, [1, 0]]
    # return pixels

    M = copy.copy(cumulativeEnergyMap)
    pixels = np.array([[np.argmin(M[:, -1]), M.shape[1] - 1]])

    y = pixels[0, 1]
    while y > 0:
        x = pixels[0, 0]
        s_min, s_argmin = np.inf, [-1, -1]
        # assert y >= 0 and y < M.shape[1], f"y has become {y} at row {x} where as image size is {M.shape}"

        if x - 1 >= 0 and M[x - 1, y - 1] <= s_min: 
            s_min, s_argmin = M[x - 1, y - 1], [x - 1, y - 1]
        
        if M[x, y - 1] <= s_min: s_min, s_argmin = M[x, y - 1], [x, y - 1]
        
        if x + 1 < M.shape[0] and M[x + 1, y - 1] <= s_min: 
            s_min, s_argmin = M[x + 1, y - 1], [x + 1, y - 1]
        
        pixels = np.row_stack((np.array([s_argmin]), pixels))
        y -= 1
    # print(pixels.shape, M.shape)
    return pixels
