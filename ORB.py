## ORB

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions
from scipy import ndimage
import time

#%%
## Image preprocessing 

# choose 1 image: NORMAL2-IM-0329-0001.jpeg
filename = 'NORMAL2-IM-0329-0001.jpeg'
im = cv2.imread('images/x-ray/'+filename,0)

# divide image into two parts with 30% overlap
full,topGray,bottomGray = pre.splitY(0.3,im)
topGray = ndimage.rotate(topGray, 30)

scale_percent = 80 # percent of original size
width = int(topGray.shape[1] * scale_percent / 100)
height = int(topGray.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
topGray = cv2.resize(topGray, dim, interpolation = cv2.INTER_AREA)

start = time.time()

#%% Feature extraction and description
# ORB
# create ORB dectector
orb = cv2.ORB_create(nfeatures=3000)  # default features is 500

# find key point and descriptor
kp1, des1 = orb.detectAndCompute(topGray, None)
kp2, des2 = orb.detectAndCompute(bottomGray, None)

# draw key point on image
result_im1 = cv2.drawKeypoints(topGray, kp1, None, (255,0,0), flags=0)
result_im2 = cv2.drawKeypoints(bottomGray, kp2, None,(255,0,0), flags=0)

# Using Brute Force matcher with Hamming distance
# create BFMatcher object
distance_method = cv2.NORM_HAMMING
bf = cv2.BFMatcher(distance_method)

# Match descriptors
matches_knn = bf.knnMatch(des1,des2,k=2)

# Apply ratio test (Lowes ratio)
good_matches = []
for m,n in matches_knn:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

# Sort them in the order of their distance.
good_matches = sorted(good_matches, key = lambda x:x.distance)
# Draw first 10 matches.

best_x_matches = len(good_matches)
best_x_matches = 100
im_matched = cv2.drawMatches(topGray, kp1, bottomGray, kp2, good_matches[:best_x_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# %% Homography estimation and image stitching
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches])#.reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches])#.reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

im_stitched = cvfunctions.warpImages(bottomGray, topGray, H)

stop = time.time()

print('Ran in ' + str(stop-start))

#%% images for exam presentation
f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(im_matched)
ax.set_title('ORB matches using KNN with Hamming distance')
plt.show()

f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(im_stitched, cmap='gray')
ax.set_title('Stitched image using ORB')
plt.show()
# %% All the graphs


fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
for img, ax in zip([topGray,full,bottomGray], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img,cmap='gray')
fig.suptitle('Image split')
axs[0].set_title('Top part')
axs[1].set_title('Full image')
axs[2].set_title('Bottom part')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15,4))
axs = axs.flatten()
for img, ax in zip([result_im1,result_im2], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('ORB keypoints')
axs[0].set_title('Top')
axs[1].set_title('Bottom part')
plt.show()

f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(img3)
ax.set_title('Matched ORB keypoints (best ' + str(best_x_matches)+ ' using Hamming distance)')
f.savefig('images/orb_matches.png',bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
for img, ax in zip([bottomGray,im_stitched,topGray], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
fig.suptitle('Stitched image using ORB')
axs[0].set_title('Bottom')
axs[1].set_title('Stitched image')
axs[2].set_title('Top')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15,6))
axs = axs.flatten()
for img, ax in zip([im_stitched,full], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
fig.suptitle('Stitched image using ORB')
axs[0].set_title('Stitched image')
axs[1].set_title('Original')
plt.show()
# %%
