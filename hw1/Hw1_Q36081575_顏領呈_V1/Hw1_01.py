import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2
import argparse
print("\n")

'''
Q1
'''
print("--------Q1's Answer--------")
#Q1.1
print("Q1.1-Load Image File Answer: ")
print("Show the height and width of the image in console mode.")
img = cv2.imread("Uncle_Roger.jpg")
cv2.imshow("My image", img)
height, width = img.shape[:2]
print("height =", height, "width =", width)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q1.2
print("Q1.2-Color Separation Answer: ")
print("One by one to show result images.")
img = cv2.imread("Flower.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blue = img_rgb.copy()
blue[:, :, 1] = 0
blue[:, :, 2] = 0
cv2.imshow("My image", blue)
cv2.waitKey(0)
cv2.destroyAllWindows()
green = img_rgb.copy()
green[:, :, 0] = 0
green[:, :, 2] = 0
cv2.imshow("My image", green)
cv2.waitKey(0)
cv2.destroyAllWindows()
red = img_rgb.copy()
red[:, :, 0] = 0
red[:, :, 1] = 0
cv2.imshow("My image", red)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q1.3
print("Q1.3-Image Flipping Answer: ")
print("Flip the image and open a new window to show the result.")
img = cv2.imread("Uncle_Roger.jpg")
flippedImg = cv2.flip(img, 1)
cv2.imshow("My image", flippedImg)
cv2.waitKey(0)

#Q1.4
print("Q1.4-Blending Answer: ")
print("Use Trackbar to change the weights and show the result in the new window.")
img1 = cv2.imread("Uncle_Roger.jpg")
img2 = cv2.flip(img1, 1)
cv2.imwrite('filpUncle.jpg', img2)
alpha_slider_max = 100
title_window = 'Linear Blend'
def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    cv2.imshow(title_window, dst)
parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input1', help='Path to the first input image.', default='Uncle_Roger.jpg')
parser.add_argument('--input2', help='Path to the second input image.', default='filpUncle.jpg')
args = parser.parse_args()
src1 = cv2.imread(cv2.samples.findFile(args.input1))
src2 = cv2.imread(cv2.samples.findFile(args.input2))
if src1 is None:
    print('Could not open or find the image: ', args.input1)
    exit(0)
if src2 is None:
    print('Could not open or find the image: ', args.input2)
    exit(0)
cv2.namedWindow(title_window)
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
on_trackbar(0)
cv2.waitKey()