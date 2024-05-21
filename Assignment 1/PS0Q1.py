import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy


A = np.load('inputAPS0Q1.npy')

# Q1: Intensity
intensities = np.sort(A.reshape(-1))[::-1]
plt.plot(intensities)
plt.xlabel('Index')
plt.ylabel('Intensity')
plt.title('Intensities in a sorted decreasing order')
plt.show()

#Q2: Histogram
plt.hist(intensities, bins=20)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Intensities')
plt.show()

#Q3: Left bottom quadrant
X = A[50:, :50]
np.save('outputXPS0Q1.npy', X)

#Q4: Subtracting mean intensity value
Y = cv2.subtract(A, A.mean())
np.save('outputYPS0Q1.npy', Y)

#Q5: Coloring red after a certain mean of A (here Y is A - A.mean(), so Y is taken)
Z = copy.deepcopy(Y)
Z[np.where(Z != 0)] = 255
Z = np.array([Z, np.zeros((100,100)), np.zeros((100,100))]).T
Z = np.uint(Z)
plt.imshow(Z, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('outputzPS0Q1.png', bbox_inches='tight', pad_inches=0)
plt.close()