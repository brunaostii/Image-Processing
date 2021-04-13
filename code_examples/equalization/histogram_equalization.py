import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('cat_before.png', 0)

# show original image
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# calculate histogram
hist, bins = np.histogram(img.flatten(), 256, [0,256])

# calculate cumulative distribution function (cdf)

cdf = hist.cumsum()
cdf_normalized = (cdf * hist.max()) / cdf.max()

# show histogram and cdf
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0,256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

# compute equalized image through opencv
equ1 = cv2.equalizeHist(img)

# show equalized image
plt.imshow(equ1, cmap='gray', vmin=0, vmax=255)
plt.show()

# compute equalized image through cdf
transf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
transf = transf.astype('uint8')

equ2 = transf[img]

# show equalized image
plt.imshow(equ2, cmap='gray', vmin=0, vmax=255)
plt.show()

# show histogram after equalization
plt.hist(equ1.flatten(), 256, [0,256], color='r')
plt.xlim([0, 256])
plt.show()

