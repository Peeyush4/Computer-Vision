import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

#Reading image
img = cv2.imread('inputPS0Q2.jpg')

#Q1: Swapping red and green
red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
swapImgPS0Q2 = np.matmul(img, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
plt.imshow(swapImgPS0Q2)
plt.axis('off')
plt.savefig('swapImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#Q2: Grey image
grayImgPS0Q2 = np.dot(img, [0.2989, 0.5870, 0.1140])
plt.imshow(grayImgPS0Q2, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('grayImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#Q3.a: Negative Image
negativeImgPS0Q2 = 256 - grayImgPS0Q2
plt.imshow(negativeImgPS0Q2, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('negativeImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#Q3.b: Mirror Image
mirrorImgPS0Q2 = grayImgPS0Q2[:, ::-1]
plt.imshow(mirrorImgPS0Q2, cmap=plt.get_cmap('grey'))
plt.axis('off')
plt.savefig('mirrorImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#Q3.c: Average of original and its mirror
avgImgPS0Q2 = np.uint8((grayImgPS0Q2 + mirrorImgPS0Q2)/2)
plt.imshow(avgImgPS0Q2, cmap=plt.get_cmap('grey'))
plt.axis('off')
plt.savefig('avgImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#Q3.d: Adding noise to the image
noise = np.random.randint(low=0, high=256, size=grayImgPS0Q2.shape)
np.save('noise.npy', noise)
addNoiseImgPS0Q2 = cv2.add(np.uint8(grayImgPS0Q2), np.uint8(noise))
plt.imshow(addNoiseImgPS0Q2, cmap=plt.get_cmap('grey'))
plt.axis('off')
plt.savefig('addNoiseImgPS0Q2.png', bbox_inches='tight', pad_inches=0)
plt.close()

#All plots
rows, columns = 3, 2
fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(10,10))
images = [swapImgPS0Q2, grayImgPS0Q2, negativeImgPS0Q2, mirrorImgPS0Q2, avgImgPS0Q2, addNoiseImgPS0Q2]
titles = ['Image after swapping red and green', 'Gray image', 'Negative Image', 
          'Image mirrored left to right', 'Average for grey image and its mirror',
          'Addition of noise in the image']
for r in range(rows):
    for c in range(columns):
        ax[r, c].imshow(images[r * columns + c], cmap=plt.get_cmap('gray'))
        ax[r, c].axis('off')
        ax[r, c].set_title(titles[r * columns + c])
fig.suptitle("Image Modifications")
plt.show()