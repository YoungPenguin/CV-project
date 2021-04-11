### CV project 

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre

#%% pre

## Image preprocessing 
# choose 1 image: NORMAL2-IM-0329-0001.jpeg
# divide image into two parts with 30% overlap, use open cv

filename = 'NORMAL2-IM-0329-0001.jpeg'
im = cv2.imread('images/x-ray/'+filename)

full,top,bottom = pre.splitY(0.3,im)

fig, ax = plt.subplots(1,3)
ax[0].imshow(top)
ax[1].imshow(full)
ax[2].imshow(bottom)

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








