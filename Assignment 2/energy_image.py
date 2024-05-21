import numpy as np
import scipy as sp
import cv2

def image_gradient_x_axis(gray_image):
    H = np.array([[1, -1]])
    gradient_x_axis = sp.ndimage.convolve(gray_image, H, cval=0, mode='constant')
    return gradient_x_axis

def image_gradient_y_axis(gray_image):
    H = np.array([[1], [-1]])
    gradient_y_axis = sp.ndimage.convolve(gray_image, H, cval=0, mode='constant')
    return gradient_y_axis

def energy_image(im):
    #Input im: Image with MxNx3 dimensions
    #Output: Return 2D energy matrix of size MxN
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gradient_x_axis = image_gradient_x_axis(gray_image)
    gradient_y_axis = image_gradient_y_axis(gray_image)
    return np.abs(gradient_x_axis) + np.abs(gradient_y_axis)
