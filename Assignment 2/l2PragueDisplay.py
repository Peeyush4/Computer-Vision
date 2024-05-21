from reduceWidth import reduceWidth
from reduceHeight import reduceHeight
from l2_energy_image import l2_energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
import cv2
import matplotlib.pyplot as plt
import tqdm

PragueImage = cv2.imread('inputSeamCarvingPrague.jpg', cv2.IMREAD_UNCHANGED)
PragueEnergy = l2_energy_image(PragueImage)
plt.imshow(PragueEnergy, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('l2PragueEnergy.png', bbox_inches='tight', pad_inches=0)


PragueHorizontalMap = cumulative_minimum_energy_map(PragueEnergy, 'HORIZONTAL')
plt.imshow(PragueHorizontalMap)
plt.axis('off')
plt.savefig('l2PragueHorizontal.png', bbox_inches='tight', pad_inches=0)


PragueVerticalMap = cumulative_minimum_energy_map(PragueEnergy, 'VERTICAL')
plt.imshow(PragueVerticalMap)
plt.axis('off')
plt.savefig('l2PragueVertical.png', bbox_inches='tight', pad_inches=0)


PragueReducedWidth, _ = reduceWidth(im=PragueImage, energyImage=PragueEnergy)
cv2.imwrite('output1L2ReducedWidthPrague.png', PragueReducedWidth)


PragueReducedHeight, _ = reduceHeight(im=PragueImage, energyImage=PragueEnergy)
cv2.imwrite('output1L2ReducedHeightPrague.png', PragueReducedHeight)


