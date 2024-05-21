import numpy as np
import scipy as sp
import copy

def cumulative_minimum_energy_map(energyImage, seamDirection):
    #Input energyImage: Energy matrix of size NxM
    #Input seamDirection: direction of the seam. "HORIZONTAL" or "VERTICAL"
    #Output: seam matrix
    Energy = copy.copy(energyImage)

    if seamDirection == 'HORIZONTAL':
        #Energy matrix
        M = np.array([Energy[0, :]])
        for i in range(1, Energy.shape[0]):
            E = Energy[i, :]
            M_middle = M[i - 1, :]
            M_ltor = np.append(np.array([np.inf]), M_middle[:-1])
            M_rtol = np.append(M_middle[1:], np.array([np.inf]))
            M_min = np.minimum(M_middle, M_ltor)
            M_min = np.minimum(M_min, M_rtol)
            M = np.row_stack((M, E + M_min))
        return M
    
    # Energy = np.rot90(energyImage, 1) #Rotate 90 degrees anticlockwise
    # M = np.rot90(M, -1) #Rotate 90 degrees clockwise
            #Energy matrix
    M = np.array([Energy[:, 0]])
    for i in range(1, Energy.shape[1]):
        E = Energy[:, i]
        M_middle = M[i - 1, :]
        M_ltor = np.append(np.array([np.inf]), M_middle[:-1])
        M_rtol = np.append(M_middle[1:], np.array([np.inf]))
        M_min = np.minimum(M_middle, M_ltor)
        M_min = np.minimum(M_min, M_rtol)
        M = np.row_stack((M, E + M_min))
    M = M.T
    return M