import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import copy

def warpImage(inputIm, refIm, H):
    # warpIm, is the input image inputIm warped 
    # according to H to fit within the frame of the reference image refIm
    #Get Boundaries 
    input_boundaries = np.array([[a, b, 1] 
                                 for a in [0, inputIm.shape[1]] 
                                 for b in [0, inputIm.shape[0]]
                                ])
    ref_boundaries = np.matmul(H, input_boundaries.T)
    ref_boundaries = (ref_boundaries / ref_boundaries[-1, :])[:2, :]
    
    #Minimum and Maximum boundaries
    min_ref = np.floor(np.min(ref_boundaries, axis=1))
    max_ref = np.ceil(np.max(ref_boundaries, axis=1))

    #Get all pixels for the warpIm dimension using boundaries
    xx, yy = np.meshgrid(np.arange(min_ref[0], max_ref[0]), 
                         np.arange(min_ref[1], max_ref[1]))
    pixels = np.row_stack((xx.reshape(-1), yy.reshape(-1), np.ones(xx.shape[0] * xx.shape[1])))

    #Convert these pixels to match the coordinates with the inputIm
    H_inv = np.linalg.inv(H)
    pixels = np.matmul(H_inv, pixels)
    pixels = pixels / pixels[-1, :]
    pixels = pixels[:-1]
    
    #Find out warpIm using the pixels
    warpIm = []
    for i in range(pixels.shape[1]):
        #x - column and y - row
        if pixels[1, i] < inputIm.shape[0] and pixels[1, i] >= 0 and \
            pixels[0, i] < inputIm.shape[1] and pixels[0, i] >= 0:
                #Pixels are in the correct InputIm dimension
                warpIm.append(inputIm[np.int16(pixels[1, i]), np.int16(pixels[0, i])])
        #Pixels are not in the correct inputIm dimension
        else: warpIm.append(np.zeros(3))
    
    warpIm = np.array(warpIm).reshape(xx.shape[0], xx.shape[1], 3).astype(np.uint8)
    
    #Merge Image
    mergeIm = copy(warpIm)
    #If there is no [0, 0] in meshgrid, we need to add columns to make sure it is present
    if xx[0, 0] * xx[0, -1] > 0:
        if xx[0, 0] < 0:
            xx2, yy2 = np.meshgrid(np.arange(xx[0][-1] + 1, 1), np.arange(min_ref[1], max_ref[1]))
            add_zeros = np.zeros((xx2.shape[0], xx2.shape[1], 3))
            xx = np.column_stack((xx, xx2))
            yy = np.column_stack((yy, yy2))
            mergeIm = np.column_stack((mergeIm, add_zeros))
        else:
            xx2, yy2 = np.meshgrid(np.arange(0, xx[0][0]), np.arange(min_ref[1], max_ref[1]))
            add_zeros = np.zeros((xx2.shape[0], xx2.shape[1], 3))
            xx = np.column_stack((xx2, xx))
            yy = np.column_stack((yy2, yy))
            mergeIm = np.column_stack((add_zeros, mergeIm))

    if yy[0, 0] * yy[-1, 0] > 0:
        if yy[0, 0] < 0:
            xx2, yy2 = np.meshgrid(np.arange(xx[0, 0], xx[0, -1] + 1), np.arange(yy[0][-1] + 1, 1))
            add_zeros = np.zeros((xx2.shape[0], xx2.shape[1], 3))
            xx = np.row_stack((xx, xx2))
            yy = np.row_stack((yy, yy2))
            mergeIm = np.row_stack((mergeIm, add_zeros))
        else:
            xx2, yy2 = np.meshgrid(np.arange(xx[0, 0], xx[0, -1] + 1), np.arange(0, yy[0][0]))
            add_zeros = np.zeros((xx2.shape[0], xx2.shape[1], 3))
            xx = np.row_stack((xx2, xx))
            yy = np.row_stack((yy2, yy))
            mergeIm = np.row_stack((add_zeros, mergeIm))

    #Since we have [0, 0] in meshgrid, we need to find this origin
    refStart_x, refStart_y = np.where(xx[0] == 0)[0][0], np.where(yy[:, 0] == 0)[0][0] 
    
    #If there are lesser dimensions in warpIm than in refIm from the origin point in warpIm, 
    #add rows and columns to make it to the correct precision
    dimension_x =  mergeIm.shape[1] - refStart_x
    if dimension_x < refIm.shape[1]: #Increasing columns
        mergeIm = np.column_stack((mergeIm, 
                                   np.zeros((mergeIm.shape[0], refIm.shape[1] - dimension_x + 1, 3))))

    dimension_y =  mergeIm.shape[0] - refStart_y
    if dimension_y < refIm.shape[0]: #Increasing rows
        mergeIm = np.row_stack((mergeIm, 
                                np.zeros((refIm.shape[0] - dimension_y + 1, mergeIm.shape[1], 3))))  
    
    #Add refIm to warpIm
    for i in range(refStart_y, refIm.shape[0] + refStart_y):
        for j in range(refStart_x, refIm.shape[1] + refStart_x):
            if (mergeIm[i, j] == np.zeros(3)).all():  
                mergeIm[i, j] = refIm[i - refStart_y, j - refStart_x]
    
    return warpIm, mergeIm
