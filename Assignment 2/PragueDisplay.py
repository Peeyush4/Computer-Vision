from reduceWidth import reduceWidth
from reduceHeight import reduceHeight
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
import cv2
import matplotlib.pyplot as plt
import tqdm
from displaySeam import displaySeam
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from find_optimal_vertical_seam import find_optimal_vertical_seam

PragueImage = cv2.imread('inputSeamCarvingPrague.jpg', cv2.IMREAD_UNCHANGED)
PragueEnergy = energy_image(PragueImage)
plt.imshow(PragueEnergy, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('PragueEnergy.png', bbox_inches='tight', pad_inches=0)


PragueHorizontalMap = cumulative_minimum_energy_map(PragueEnergy, 'HORIZONTAL')
plt.imshow(PragueHorizontalMap)
plt.axis('off')
plt.savefig('PragueHorizontal.png', bbox_inches='tight', pad_inches=0)


PragueVerticalMap = cumulative_minimum_energy_map(PragueEnergy, 'VERTICAL')
plt.imshow(PragueVerticalMap)
plt.axis('off')
plt.savefig('PragueVertical.png', bbox_inches='tight', pad_inches=0)


horizontalSeam = find_optimal_horizontal_seam(cumulativeEnergyMap=PragueHorizontalMap)
HorizontalImageSeam = displaySeam(im=PragueImage, seam=horizontalSeam, type='HORIZONTAL')
cv2.imwrite('PragueHorizontalSeam.png', HorizontalImageSeam)


VerticalSeam = find_optimal_vertical_seam(cumulativeEnergyMap=PragueVerticalMap)
VerticalImageSeam = displaySeam(im=PragueImage, seam=VerticalSeam, type='VERTICAL')
cv2.imwrite('PragueVerticalSeam.png', VerticalImageSeam)


PragueReducedHeight, _ = reduceHeight(im=PragueImage, energyImage=PragueEnergy)
cv2.imwrite('output1ReducedHeightPrague.png', PragueReducedHeight)

PragueReducedWidth, _ = reduceWidth(im=PragueImage, energyImage=PragueEnergy)
cv2.imwrite('output1ReducedWidthPrague.png', PragueReducedWidth)

