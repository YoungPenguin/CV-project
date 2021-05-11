## SIFT

#%% imports

import cv2
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions
from scipy import ndimage
import time

#%% Image preprocessing 
#cv2.destroyAllWindows() 

# choose 1 image: NORMAL2-IM-0329-0001.jpeg
filename = 'NORMAL2-IM-0329-0001.jpeg'
im = cv2.imread('images/x-ray/'+filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to gray scale

#topGray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
#bottomGray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

# divide image into two parts with 30% overlap
full,top,bottom = pre.splitY(0.3,im)

# transforming top image
top = ndimage.rotate(top, 30) # rotating 
scale_percent = 80 # percent of original size
width = int(top.shape[1] * scale_percent / 100)
height = int(top.shape[0] * scale_percent / 100)
dim = (width, height)
top = cv2.resize(top, dim, interpolation = cv2.INTER_AREA) # scale/resize image

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
for img, ax in zip([full,top,bottom], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
fig.suptitle('Image split')
axs[0].set_title('Full image')
axs[0].axis('off')
axs[1].set_title('Top part')
axs[1].axis('off')
axs[2].set_title('Bottom part')
axs[2].axis('off')
plt.show()

topGray = top
bottomGray = bottom

start = time.time()

#%% Feature decetion and description
# Initiate SIFT detector 
sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.03, sigma = 1.6, edgeThreshold = 10)
#sift = cv2.SIFT_create()
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

good_match = sorted(good_match, key = lambda x:x.distance)
        
im_matched = cv2.drawMatches(topGray, kp1, bottomGray, kp2, good_match[:100],None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imshow('SIFT matches', im_matched)
#%% Image sticting using Homography matrix 
#cv2.destroyAllWindows() 
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

h,w = topGray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,H)

im_stitched = cvfunctions.warpImages(bottomGray, topGray, H)
#plt.imshow(im_stitched, cmap='gray')
#plt.show()

stop = time.time()

print('Ran in ' + str(stop-start))

f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(im_stitched, cmap='gray')
ax.set_title('Matched SIFT keypoints')
plt.show()

#%% OLD
'''
# Blob detection
plt.close('all')
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