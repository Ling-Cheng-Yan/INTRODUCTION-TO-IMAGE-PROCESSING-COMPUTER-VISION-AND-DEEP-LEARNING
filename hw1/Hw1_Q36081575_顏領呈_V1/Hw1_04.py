import cv2
import numpy as np

'''
Q4
'''
print("--------Q4's Answer--------")
#Q4.1
print("Q4.1-Transforms: Rotation, Scaling, Translation Answer: ")
print("Please rotate, scale and translate the yellow parrot (as image below) with following parameters.")
img = cv2.imread('Parrot.png')
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
resized = cv2.resize(rotated, (540, 810))
dimensions = img.shape
ratio = 500.0 / w
dim = (500, int(h * ratio))
resized_2 = cv2.resize(rotated, dim)
height, width = resized_2.shape[:2]
M = np.float32([[1, 0, 200], [0, 1, 5]])
translated = cv2.warpAffine(resized_2, M, (width, height))
cv2.imshow('My Transforms Image', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()