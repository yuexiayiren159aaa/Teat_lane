import cv2
import matplotlib.pyplot as plt
import numpy as np

imgfile = 'datasets/PreliminaryData/testB/7.jpg'
pngfile = 'results/result/7.png'

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

img = img[:, :, ::-1]
img[..., 2] = np.where(mask == 1, 255, img[..., 2])

# plt.imshow(img)
# plt.show()

cv2.imwrite("7.jpg",img)