import numpy as np
import scipy as sp

def image_gradient_x_axis(gray_image):
    H = np.array([[1, -1]])
    gradient_x_axis = sp.ndimage.convolve(gray_image, H, cval=0, mode='constant')
    return gradient_x_axis

def image_gradient_y_axis(gray_image):
    H = np.array([[1], [-1]])
    gradient_y_axis = sp.ndimage.convolve(gray_image, H, cval=0, mode='constant')
    return gradient_y_axis

def l2_energy_image(im):
    #Input im: Image with MxNx3 dimensions
    #Output: Return 2D energy matrix of size MxN
    gray_image = np.dot(im, [0.2989, 0.5870, 0.1140])
    gradient_x_axis = image_gradient_x_axis(gray_image)
    gradient_y_axis = image_gradient_y_axis(gray_image)
    return np.sqrt(np.square(gradient_x_axis) + np.square(gradient_y_axis))
