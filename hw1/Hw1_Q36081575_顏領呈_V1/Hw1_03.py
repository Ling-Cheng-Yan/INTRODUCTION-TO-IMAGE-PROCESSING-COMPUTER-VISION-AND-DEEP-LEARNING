import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from scipy.ndimage.filters import generic_filter

print("\n")

'''
Q3
'''
print("--------Q3's Answer--------")
#Q3.1
print("Q3.1-Gaussian Blur Answer: ")
print("Convert the color image into a grayscale image, then smooth it by your own 3x3 Gaussian smoothing filter.")
#3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2))
#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()

image = cv2.imread("Chihiro.jpg")
#灰階
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
#卷積
grad = signal.convolve2d(gray, gaussian_kernel, boundary='symm', mode='same')
plt.imshow(grad, cmap=plt.get_cmap('gray'))
plt.show()
cv2.imwrite('gaussian_Chihiro.jpg', grad)


#Q3.2
print("Q3.2-Sobel X Answer: ")
print("Use Sobel edge detection to detect vertical edge by your own 3x3 Sobel X operator.")
'''
def sobelOperatorX(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container
    pass

b_sobelx = cv2.cvtColor(cv2.imread("gaussian_Chihiro.jpg"), cv2.COLOR_BGR2GRAY)
b_sobelx = sobelOperatorX(b_sobelx)
b_sobelx = cv2.cvtColor(b_sobelx, cv2.COLOR_GRAY2RGB)
plt.imshow(b_sobelx)
plt.show()
cv2.imwrite('SobelX_Chihiro.jpg', b_sobelx)
'''
def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

img = cv2.imread('Chihiro.jpg', 0)
blurred_img = cv2.imread('gaussian_Chihiro.jpg', 0)

s_mask = 17

sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=s_mask))
sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
b_sobely = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=s_mask))
b_sobely = interval_mapping(b_sobely, np.min(sobely), np.max(sobely), 0, 255)

fig = plt.figure(figsize=(10, 14))

plt.subplot(3,2,4),plt.imshow(b_sobely,cmap = 'gray')
plt.title('Blurred Sobel X'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()
cv2.imwrite('SobelY_Chihiro.jpg', b_sobely)


#Q3.3
print("Q3.3-Sobel Y Answer: ")
print("Use Sobel edge detection to detect horizontal edge by your own 3x3 Sobel Y operator.")
def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

img = cv2.imread('Chihiro.jpg', 0)
blurred_img = cv2.imread('gaussian_Chihiro.jpg', 0)

s_mask = 17

sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=s_mask))
sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
b_sobely = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=s_mask))
b_sobely = interval_mapping(b_sobely, np.min(sobely), np.max(sobely), 0, 255)

fig = plt.figure(figsize=(10, 14))

plt.subplot(3,2,4),plt.imshow(b_sobely,cmap = 'gray')
plt.title('Blurred Sobel Y'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()
cv2.imwrite('SobelY_Chihiro.jpg', b_sobely)


#Q3.4
print("Q3.4-Magnitude Answer: ")
print("Use the results of 3.2) Sobel X and 3.3) Sobel Y to calculate the magnitude.")
def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

img = cv2.imread('gaussian_Chihiro.jpg')
mag = getGradientMagnitude(img)
cv2.imshow('My Magnitude Image', mag)
cv2.waitKey(0)
cv2.destroyAllWindows()