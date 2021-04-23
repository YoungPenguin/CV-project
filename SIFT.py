## SIFT

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions

#%%
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
plt.show()
#%% Feature detection
# Set parametes
sigma = 1.6
n = 3 # n ~ 3
thres = 0.01 # thres ~ 0.01

blobsTop = cvfunctions.detectBlobs(topGray/255.0, sigma, n, thres)
blobsBot = cvfunctions.detectBlobs(bottomGray/255.0, sigma, n, thres)

# Show blobs
fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
ax[0].imshow(topGray,cmap='gray')
ax[0].scatter(blobsTop[0,:], blobsTop[1,:],s=25,c='r',marker='.')
ax[1].imshow(bottomGray,cmap='gray')
ax[1].scatter(blobsBot[0,:], blobsBot[1,:],s=25,c='r',marker='.')
plt.title = 'Blobs'
plt.show()

#%% Feature description
# Initiate SIFT detector 
sift = cv2.SIFT_create()
# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(topGray,None) 
kp2, des2 = sift.detectAndCompute(bottomGray,None)

#%% Feature matching
# Initiate BFMatcher with default params
bf = cv2.BFMatcher() 
# Find matches with KNN (k-nearest neighbours)
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test (Lowes ratio)
good_match = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_match.append(m)

# Draw (good) matches (use either drawMatchesKnn with good_match or drawMatches with good_match_flat)
#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   flags = 2)

im_matched = cv2.drawMatches(topGray, kp1, bottomGray, kp2, good_match,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#im_matches = cv2.drawMatches(topGray, kp1, bottomGray, kp2, good_match,None,**draw_params)
cv2.imshow('SIFT matches', im_matched)
        
#%% Image sticting using Homography matrix 
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

h,w = topGray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,H)

im_stitched = cvfunctions.warpImages(bottomGray, topGray, H)
plt.imshow(im_stitched, cmap='gray')
plt.show()

#%% EXTRA
'''
matchesMask = mask.ravel().tolist()
h,w = topGray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,H)
bottomGray_copy = bottomGray.copy()
bottomGrayLine = cv2.polylines(bottomGray_copy,[np.int32(dst)],True,255,3, cv2.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
im_matches = cv2.drawMatches(topGray,kp1,bottomGrayLine,kp2,good_match,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('SIFT matches', im_matches)
'''