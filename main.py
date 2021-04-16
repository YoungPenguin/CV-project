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
# create ORB dectector
orb = cv2.ORB_create(nfeatures=3000)  # default features is 500

# find key point and descriptor
kp1, des1 = orb.detectAndCompute(topGray, None)
kp2, des2 = orb.detectAndCompute(bottomGray, None)

# draw key point on image
result_im1 = cv2.drawKeypoints(topGray, kp1, None, flags=0)
result_im2 = cv2.drawKeypoints(bottomGray, kp2, None, flags=0)

#cv2.imshow("Top image",result_im1)
#cv2.imshow("Bottom image",result_im2)
#cv2.waitKey(0)

# FLANN parameters
flann_index_lsh = 6
index_params = dict(algorithm=flann_index_lsh,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2)
search_params = dict(checks=100)  # or pass empty dictionary

# create FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)

# perform matching
flann_matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matches_mask = [[0, 0] for i in range(len(flann_matches))]

# ratio test as per Lowe's paper
good = []
for index in range(len(flann_matches)):
    if len(flann_matches[index]) == 2:
        m, n = flann_matches[index]
        if m.distance < 0.8 * n.distance:  # 0.8 is threshold of ratio testing
            matches_mask[index] = [1, 0]
            good.append(flann_matches[index])

draw_params = dict(
    singlePointColor=(255, 0, 0),
    matchesMask=matches_mask,
    flags=2)

img3 = cv2.drawMatchesKnn(topGray, kp1, bottomGray, kp2, flann_matches, None, **draw_params)
cv2.imshow("Final image",img3)
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

# Find homography

flattened = [val for sublist in good for val in sublist]

src_pts = np.float32([ kp1[m.queryIdx].pt for m in flattened ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in flattened ]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w = topGray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,H)
bottomGrayLine = bottomGray.copy()
bottomGrayLine = cv2.polylines(bottomGrayLine ,[np.int32(dst)],True,255,3, cv2.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(topGray,kp1,bottomGrayLine,kp2,flattened,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

#H = cvfunctions.findHomography(src_pts, dst_pts)

# Stitching
output = cvfunctions.warpImages(bottomGray, topGray, H)
plt.imshow(output)
plt.show()

#%% Harris corners (week 6)
# convert to gray-scale
s = 15 # s = [15;50]
eps = 1.0 # eps > 0.5
k = 0.1 # k < 0.2
tau = 0 # tau ~ 0 ([-0.5;0.5])

ctop = cvfunctions.cornerDetector(topGray/255.0, s, eps, k, tau)
cbot = cvfunctions.cornerDetector(bottomGray/255.0, s, eps, k, tau)

# plot image and corners
fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
ax[0].imshow(topGray,cmap='gray')
ax[0].scatter(ctop[0,:], ctop[1,:],s=25,c='r',marker='.')
ax[1].imshow(bottomGray,cmap='gray')
ax[1].scatter(cbot[0,:], cbot[1,:],s=25,c='r',marker='.')
plt.show()

descriptTop, pts_top = cvfunctions.simpleDescriptor(topGray, ctop, 5)
descriptBot, pts_bot = cvfunctions.simpleDescriptor(bottomGray, cbot, 5)

#%%

## Feature matching
# Use Hamming distance for ORB

## Homography estimation (week 5)
# Using SVD

## Image stitching
# Using homography matrix





