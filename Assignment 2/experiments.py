from reduceWidth import reduceWidth
from reduceHeight import reduceHeight
from energy_image import energy_image
from l2_energy_image import l2_energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from find_optimal_vertical_seam import find_optimal_vertical_seam
from displaySeam import displaySeam

import cv2
import matplotlib.pyplot as plt
import tqdm
import argparse
import copy

parser = argparse.ArgumentParser(description='Options for running this script')
parser.add_argument('-i', '--image', default='inputSearmCarvingPrague.jpg', help="Image file name you want to use. Default is inputSearmCarvingPrague.jpg")
parser.add_argument('-e', '--energy', default='l1', help="Energy you want to use. Default is l1")
parser.add_argument('-hs', '--horizontalSeam', default=1, type=int, help="Creates horizontal seam. Reduction in width.")
parser.add_argument('-vs', '--verticalSeam', default=1, type=int, help="Creates vertical seam. Reduction in height.")
parser.add_argument('-b', '--both', action='store_true', help="Creates horizontal and vertical seam simultaneously")
# parser.add_argument('--sampling', default=False, help='There are 2 options "Random" and "SMOTE" (no casing restriction), which uses Imbalanced libraries')
# parser.add_argument('--sampling_random_seed', default=None, help='If sampling is chosen, then the seed for it')
# parser.add_argument('--senate', default='Senate', help="If all, type 'all', else type 'House'")
args = parser.parse_args()

Image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)

#Resized image
dim = (Image.shape[1] - args.horizontalSeam, Image.shape[0] - args.verticalSeam)
resized_image = cv2.resize(Image, dim)
cv2.imwrite(f'{args.image.split(".")[0]}ResizedImage-{dim}.png', resized_image)


Energy = energy_image(Image)
plt.imshow(Energy, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig(f'{args.image.split(".")[0]}Energy-{dim}.png', bbox_inches='tight', pad_inches=0)


if args.energy == 'l2':
    Energy = l2_energy_image(Image)
plt.imshow(Energy, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig(f'{args.image.split(".")[0]}Energy{args.energy}-{dim}.png', bbox_inches='tight', pad_inches=0)


HorizontalMap = cumulative_minimum_energy_map(Energy, 'HORIZONTAL')
plt.imshow(HorizontalMap)
plt.axis('off')
plt.savefig(f'{args.image.split(".")[0]}Horizontal{args.energy}-{dim}.png', bbox_inches='tight', pad_inches=0)


VerticalMap = cumulative_minimum_energy_map(Energy, 'VERTICAL')
plt.imshow(VerticalMap)
plt.axis('off')
plt.savefig(f'{args.image.split(".")[0]}Vertical{args.energy}-{dim}.png', bbox_inches='tight', pad_inches=0)


hSeam = find_optimal_horizontal_seam(HorizontalMap)
HorizontalSeam = displaySeam(Image, hSeam, 'HORIZONTAL')
cv2.imwrite(f'{args.image.split(".")[0]}HorizontalSeam{args.energy}-{dim}.png', HorizontalSeam)


vSeam = find_optimal_vertical_seam(VerticalMap)
VerticalSeam = displaySeam(Image, vSeam, 'VERTICAL')
cv2.imwrite(f'{args.image.split(".")[0]}VerticalSeam{args.energy}-{dim}.png', VerticalSeam)


im, en = copy.copy(Image), copy.copy(Energy)
for _ in range(args.verticalSeam):
    im, en = reduceHeight(im=im, energyImage=en)
cv2.imwrite(f'{args.image.split(".")[0]}output1ReducedHeight{args.energy}-{args.verticalSeam}-{dim}.png', im)

im, en = copy.copy(Image), copy.copy(Energy)
for _ in range(args.horizontalSeam):
    im, en = reduceWidth(im=im, energyImage=en)
cv2.imwrite(f'{args.image.split(".")[0]}output1ReducedWidth{args.energy}-{args.horizontalSeam}-{dim}.png', im)

if args.both:
    im, en = copy.copy(Image), copy.copy(Energy)    
    for _ in range(args.verticalSeam):
        im, en = reduceHeight(im=im, energyImage=en)
    
    for _ in range(args.horizontalSeam):
        im, en = reduceWidth(im=im, energyImage=en)
    cv2.imwrite(f'{args.image.split(".")[0]}outputReduced{args.energy}-{dim}.png', im)

