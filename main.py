### CV project 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

## Image preprocessing 
# choose 1 image: NORMAL2-IM-0329-0001.jpeg
# divide image into two parts with 30% overlap, use open cv
filename = 'NORMAL2-IM-0329-0001.jpeg'
im = cv2.imread('images/x-ray/'+filename)
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float)/255.0
plt.imshow(im,cmap='gray')
plt.show()

r, c = im.shape
print(r," c:",c) # r: 1953  c: 2216

## Feature extraction and description
# ORB
# SIFT - Use Difference of Gaussians (DoG) (week 7)
# Harris corners (week 6)

## Feature matching
# Use Hamming distance for ORB

## Homography estimation (week 5)
# Using SVD

## Image stitching
# Using homography matrix








