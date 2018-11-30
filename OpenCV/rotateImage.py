import numpy as np
import cv2 as cv

# ---- Read image
img = cv.imread('cat1.jpeg')
print(img.shape)

# ---- Show the origima image
cv.imshow("original", img)
cv.waitKey(0)

# ---- Move weight 100, height 25: Height, Weight:
height, weight = img.shape[:2]
print(height, weight)
M = np.float32([[1,0,100], [0,1,25]])
img_translation = cv.warpAffine(img, M, (weight,height))
# ---- Save image
cv.imwrite('cat1_trans.png', img_translation)

# ---- Rotate Image
M = cv.getRotationMatrix2D(((weight-1)/2, (weight-1)/2), 45, 1)
img_rotation = cv.warpAffine(img, M, (weight, height))
cv.imwrite('cat1_rotate.png', img_rotation)


