import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def detectCircles(im, radius, useGradient, bin_size=3, accumulator_threshold=28, 
                  pre_threshold1=0, pre_threshold2=255, heading=None):
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    smoothed_image = cv2.medianBlur(gray_image, 3) 

    if useGradient: #Sobel operator
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        
        grad_x = cv2.Sobel(gray_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
 
        smoothed_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        smoothed_image[smoothed_image < pre_threshold1] = 0
        smoothed_image[smoothed_image > pre_threshold2] = 0
    else: #Use canny edge detection       
        smoothed_image = cv2.Canny(smoothed_image, threshold1=pre_threshold1, threshold2=pre_threshold2)
    
    # Apply binary and invert image
    _, binary_image = cv2.threshold(smoothed_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_image = cv2.bitwise_not(binary_image)
    # cv2.imshow(f"Inverted image after preprocessing for {heading}", inverted_image)
    plt.imshow(inverted_image, cmap='gray')
    plt.title(f"Inverted image after preprocessing for {heading}")
    plt.axis('off')
    plt.show()

    # Develop initial code for accumulator array
    circle_image = np.zeros(inverted_image.shape, dtype=int)
    accumulator_image = np.zeros(inverted_image.shape)
    center_y, center_x = np.where(inverted_image != 255)
    
    # Draw accumulator array
    for x, y in zip(center_x, center_y):
        cv2.circle(circle_image, (x, y), radius, (1, 1, 1), bin_size)
        accumulator_image += circle_image
        circle_image = np.zeros(inverted_image.shape)
    
    #Remove circles that do not constitute much 
    image_center_y, image_center_x = np.where(accumulator_image > accumulator_threshold)
    
    #Display accumulator image
    plt.imshow(accumulator_image)
    plt.title(f"Accumulator for {heading}")
    plt.axis('off')
    plt.show()
    # print(np.max(accumulator_image))

    #Return centers for circles
    return np.array([image_center_x, image_center_y]).T




images = [cv2.imread('eyes_deer.jpg'), cv2.imread('eyes_deer.jpg'), 
        cv2.imread('sports_balls.jpg'), cv2.imread('sports_balls.jpg')]
radii = [3, 3, 25, 25]
circles_array = [
    # Without using gradient
    detectCircles(im=images[1], radius=radii[1], useGradient=0, 
                    bin_size=1, accumulator_threshold=9,
                    pre_threshold1=0, pre_threshold2=255, 
                    heading="Deer eyes without gradient"),
    #Using gradient
    detectCircles(im=images[0], radius=radii[0], useGradient=1, 
                    bin_size=1, accumulator_threshold=0,
                    pre_threshold1=200, pre_threshold2=255,
                    heading="Deer eyes with gradient"),

    # Without using gradient    
    detectCircles(im=images[2], radius=radii[2], useGradient=0, 
                    bin_size=5, accumulator_threshold=189,
                    heading="Sports balls without gradient"),
    # Using gradient
    detectCircles(im=images[3], radius=radii[3], useGradient=1, 
                    bin_size=5, accumulator_threshold=220,
                    pre_threshold1=110, pre_threshold2=210,
                    heading="Deer eyes with gradient")
]

savefigures = ['eyes deer 3', 'eyes deer 3 gradient', 'balls size 25', 'balls size 25 gradient']

for pic in range(4):
    #Display circles 
    if circles_array[pic] is not None:
        circles = np.uint16(np.around(circles_array[pic]))
        image = images[pic]
        radius = radii[pic]

        for i in circles:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 1)
            # circle outline
            cv2.circle(image, center, radius, (255, 0, 255), 1)
            cv2.putText(
                image,
                f"R={radius}",
                (i[0] - 20, i[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        cv2.imshow("detected circles", image)
        cv2.waitKey(0)
        cv2.imwrite(f'{savefigures[pic]}.jpg', image)
