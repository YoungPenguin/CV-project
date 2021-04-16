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

# convert to gray scale
topGray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
bottomGray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1,3)
ax[0].imshow(top)
ax[1].imshow(full)
ax[2].imshow(bottom)

#%% Feature extraction and description
# ORB
#%% SIFT - Use Difference of Gaussians (DoG) (week 7)
# in order to use SIFT we use DoG to detect BLOBs. DetectBlobs does this.
# set parameters
sigma = 1.6
n = 3 # n ~ 3
thres = 0.01 # thres ~ 0.01

# find blobs
blobsTop = cvfunctions.detectBlobs(topGray/255.0, sigma, n, thres)
blobsBot = cvfunctions.detectBlobs(bottomGray/255.0, sigma, n, thres)

# show blobs
fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
ax[0].imshow(topGray,cmap='gray')
ax[0].scatter(blobsTop[0,:], blobsTop[1,:],s=25,c='r',marker='.')
ax[1].imshow(bottomGray,cmap='gray')
ax[1].scatter(blobsBot[0,:], blobsBot[1,:],s=25,c='r',marker='.')
plt.show()

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(topGray,None)
kp2, des2 = sift.detectAndCompute(bottomGray,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
plt.figure(figsize=(30,20))
img3 = cv2.drawMatchesKnn(topGray,kp1,bottomGray,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

#%% Harris corners (week 6)
# convert to gray-scale
s = 15 # s = [15;50]
eps = 1.0 # eps > 0.5
k = 0.1 # k < 0.2
tau = 0 # tau ~ 0 ([-0.5;0.5])

ctop = cvfunctions.cornerDetector(topGray, s, eps, k, tau)
cbot = cvfunctions.cornerDetector(bottomGray, s, eps, k, tau)

# plot image and corners
fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
ax[0].imshow(topGray,cmap='gray')
ax[0].scatter(ctop[0,:], ctop[1,:],s=25,c='r',marker='.')
ax[1].imshow(bottomGray,cmap='gray')
ax[1].scatter(cbot[0,:], cbot[1,:],s=25,c='r',marker='.')
plt.show()


#%%

## Feature matching
# Use Hamming distance for ORB

## Homography estimation (week 5)
# Using SVD

## Image stitching
# Using homography matrix





