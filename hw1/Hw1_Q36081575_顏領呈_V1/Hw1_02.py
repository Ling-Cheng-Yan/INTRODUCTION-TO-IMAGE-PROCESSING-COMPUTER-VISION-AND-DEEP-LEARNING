import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from skimage import io, img_as_float
from skimage.filters import gaussian
from skimage.filters import median
from skimage.morphology import disk
print("\n")

'''
Q2
'''
print("--------Q2's Answer--------")
#Q2.1
print("Q2.1-Median filter Answer: ")
print("Apply 7x7 median filter to “cat.png")
img_gaussian_noise = cv2.imread('Cat.png', 0)
img_salt_pepper_noise = cv2.imread('Cat.png', 0)

img = img_salt_pepper_noise

median_using_cv2 = cv2.medianBlur(img, 7)

median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)

cv2.imshow("Original", img)
cv2.imshow("cv2 median", median_using_cv2)
cv2.imshow("Using skimage median", median_using_skimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q2.2
print("Q2.2-Gaussian Blur Answer: ")
print("Apply 3x3 Gaussian blur to “Cat.png")
img_gaussian_noise = img_as_float(io.imread('Cat.png', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('Cat.png', as_gray=True))

img = img_gaussian_noise

gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                [1/8, 1/4, 1/8],
                [1/16, 1/8, 1/16]])

conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT) 

gaussian_using_cv2 = cv2.GaussianBlur(img, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)

cv2.imshow("Original", img)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
cv2.imshow("Using skimage gaussian", gaussian_using_skimage)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#Q2.3
print("Q2.3-Bilateral filter Answer: ")
print("Apply 9x9 Bilateral filter with 90 sigmaColor and 90 sigmaSpace to “Cat.png")
img_gaussian_noise = cv2.imread('Cat.png', 0)
img_salt_pepper_noise = cv2.imread('Cat.png', 0)

img = img_salt_pepper_noise

bilateral_using_cv2 = cv2.bilateralFilter(img, 9, 90, 90, borderType=cv2.BORDER_CONSTANT)

from skimage.restoration import denoise_bilateral
bilateral_using_skimage = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,
                multichannel=False)


cv2.imshow("Original", img)
cv2.imshow("cv2 bilateral", bilateral_using_cv2)
cv2.imshow("Using skimage bilateral", bilateral_using_skimage)

cv2.waitKey(0)
cv2.destroyAllWindows()