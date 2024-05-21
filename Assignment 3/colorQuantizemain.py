from quantizeHSV import quantizeHSV
from quantizeRGB import quantizeRGB
from getHueHists import getHueHists
from computeQuantizationError import computeQuantizationError
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('baloons.jpg')
for rgb_k in range(2, 8):
    rgbQuantizedImage, rgbCenters = quantizeRGB(image, rgb_k)
    cv2.imwrite(f'RGB_k_{rgb_k}.jpg', rgbQuantizedImage)
    print(f"SSD error for RGB Image with k={rgb_k} is {computeQuantizationError(image, rgbQuantizedImage)}")


for hsv_k in range(2, 8):
    hsvQuantizedImage, hsvCenters = quantizeHSV(image, hsv_k)
    plt.figure()
    cv2.imwrite(f'HSV_k_{hsv_k}.jpg', hsvQuantizedImage)
    plt.close()
    print(f"SSD error for HSV Image with k={hsv_k} is {computeQuantizationError(image, hsvQuantizedImage)}")
    h1, h2 = getHueHists(image, hsv_k)
    # h1.show()
    h1.savefig(f'Hue__{hsv_k}_data_histogram.jpg')
    h2.savefig(f'Hue_{hsv_k}_center_histogram.jpg')