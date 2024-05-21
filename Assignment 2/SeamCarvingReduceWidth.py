from reduceWidth import reduceWidth
from energy_image import energy_image
import cv2
import matplotlib.pyplot as plt

PragueImage = cv2.imread('inputSeamCarvingPrague.jpg', cv2.IMREAD_UNCHANGED)
PragueEnergy = energy_image(PragueImage)

for i in range(100):
    PragueImage, PragueEnergy = reduceWidth(im=PragueImage, energyImage=PragueEnergy)
cv2.imwrite('outputReduceWidthPrague.png', PragueImage)


MallImage = cv2.imread('inputSeamCarvingMall.jpg', cv2.IMREAD_UNCHANGED)
MallEnergy = energy_image(MallImage)
reducedMallImage, reducedMallEnergy = reduceWidth(im=MallImage, energyImage=MallEnergy)

for i in range(100):
    MallImage, MallEnergy = reduceWidth(im=MallImage, energyImage=MallEnergy)
cv2.imwrite('outputReduceWidthMall.png', MallImage)