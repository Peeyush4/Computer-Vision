import numpy as np
import copy

def find_optimal_vertical_seam(cumulativeEnergyMap):
    #Input cumulativeEnergyMap: np array of size MxN
    #Output: Pixels of size Mx2
    M = copy.copy(cumulativeEnergyMap)
    pixels = np.array([[M.shape[0] - 1, np.argmin(M[-1, :])]])

    x = pixels[0, 0]
    while x > 0:
        y = pixels[0, 1]
        s_min, s_argmin = np.inf, [-1, -1]
        # assert y >= 0 and y < M.shape[1], f"y has become {y} at row {x} where as image size is {M.shape}"

        if y - 1 >= 0 and M[x - 1, y - 1] <= s_min: 
            s_min, s_argmin = M[x - 1, y - 1], [x - 1, y - 1]
        
        if M[x - 1, y] <= s_min: s_min, s_argmin = M[x - 1, y], [x - 1, y]
        
        if y + 1 < M.shape[1] and M[x - 1, y + 1] <= s_min: 
            s_min, s_argmin = M[x - 1, y + 1], [x - 1, y + 1]
        
        pixels = np.row_stack((np.array([s_argmin]), pixels))
        x -= 1
    return pixels
