import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import cv2 as cv
from matchPics import matchPics

origin = cv.imread('./images/cv_cover.jpg')

x = np.arange(0, 36, 1)
y = []

for i in range(36):
    # Rotate Image
    image = rotate(origin, 10 * i, reshape=True)

    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(image, origin)

    # Update histogram
    y.append(len(matches))
    print(y)

# Display histogram
y = np.array(y)
plt.bar(x, y)
plt.show()

'''
result:
[945, 181, 35, 29, 34, 21, 16, 18, 32, 26, 27, 20, 16, 18, 20, 22, 31, 24, 30, 28, 25, 25, 17, 31, 25, 26, 19, 14, 15, 14, 24, 29, 33, 24, 37, 220]
'''
