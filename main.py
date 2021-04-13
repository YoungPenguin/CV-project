### CV project 

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions

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

#%% Feature extraction and description
# ORB
# SIFT - Use Difference of Gaussians (DoG) (week 7)

#%% Harris corners (week 6)
# convert to gray-scale
topGray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float)/255.0
bottomGray = cv2.cvtColor(bottom, cv2.COLOR_RGB2GRAY).astype(np.float)/255.0

s = 30 # s = [15;50]
eps = 1 # eps > 0.5
k = 0.1 # k < 0.2
tau = 0 # tau ~ 0 ([-0.5;0.5])

ctop = cvfunctions.cornerDetector(topGray, s, eps, k, tau)

# plot image and corners
plt.figure()
plt.imshow(top,cmap='gray')
plt.scatter(ctop[0,:], ctop[1,:],s=25,c='r',marker='.')
plt.show()

cbot = cvfunctions.cornerDetector(bottomGray, s, eps, k, tau)

# plot image and corners
plt.figure()
plt.imshow(bottom,cmap='gray')
plt.scatter(cbot[0,:], cbot[1,:],s=25,c='r',marker='.')
plt.show()

#%% SIFT - Use Difference of Gaussians (DoG) (week 7)

#%%

## Feature matching
# Use Hamming distance for ORB

## Homography estimation (week 5)
# Using SVD

## Image stitching
# Using homography matrix








