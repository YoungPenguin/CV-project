#!/usr/bin/env python3

### Harris corners

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions
from skimage.feature import corner_orientations
from skimage.morphology import octagon
from scipy import ndimage
import time

######################################################
#%% IMAGE PROCESSING
######################################################
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

#%% timing
t = time.time()

######################################################
#%% FEATURE EXTRACTION
######################################################
# find corners
s = 8 # s = [15;50]
eps = 1.3 # eps > 0.5
k = 0.06 # k ~ 0.01
tau = 0 # tau ~ 0 ([-0.5;0.5])
ctop, rtop = cvfunctions.cornerDetector(topGray/255.0, s, eps, k, tau)
cbot, rbot = cvfunctions.cornerDetector(bottomGray/255.0, s, eps, k, tau)

# plot image and corners
#fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
#ax[0].imshow(topGray,cmap='gray')
#ax[0].scatter(ctop[0,:], ctop[1,:],s=25,c='r',marker='.')
#ax[1].imshow(bottomGray,cmap='gray')
#ax[1].scatter(cbot[0,:], cbot[1,:],s=25,c='r',marker='.')
#plt.show()


######################################################
#%% FEATURE DESCRIPTOR
######################################################
# brief descriptor
(kps1, f1, kps2, f2) = cvfunctions.brief_descriptor(topGray, bottomGray, ctop, cbot)

bf = cv2.BFMatcher(cv2.NORM_HAMMING) #cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(f1,f2,k=2)

# Apply ratio test
good_match = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_match.append(m)

# Sort them in the order of their distance.
good_match = sorted(good_match, key = lambda x:x.distance)

# show matches
im_matched = cv2.drawMatches(topGray, kps1, bottomGray, kps2, good_match[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.figure(figsize=(30,20))
#cv2.imshow('Harris Corners matches', im_matched)
#plt.savefig('HarrisCorners_matches.png') 
#cv2.waitKey()
######################################################
#%% HOMOGRAPHY ESTIMATION
######################################################
# Find homography

src_pts = np.float32([ kps1[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

######################################################
#%% IMAGE STITCHING
######################################################
im_stitched = cvfunctions.warpImages(bottomGray, topGray, H)
#plt.figure()
#plt.imshow(im_stitched, cmap = 'gray')
#plt.show()


#%% images for exam presentation
f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(im_matched)
ax.set_title('Harris corner matches using KNN with Hamming distance')
plt.show()

f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(im_stitched, cmap='gray')
ax.set_title('Stitched image using Harris corners')
plt.show()

#%% elapsed time
elapsed = time.time() - t
print('The elapsed time was ', elapsed)
