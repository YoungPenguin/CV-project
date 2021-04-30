## ORB

#%% imports

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pre
import cv as cvfunctions
from scipy import ndimage

#%%
## Image preprocessing 

# choose 1 image: NORMAL2-IM-0329-0001.jpeg
filename = 'NORMAL2-IM-0329-0001.jpeg'
im = cv2.imread('images/x-ray/'+filename)

# divide image into two parts with 30% overlap
full,top,bottom = pre.splitY(0.3,im)
top = ndimage.rotate(top, 90)

scale_percent = 80 # percent of original size
width = int(top.shape[1] * scale_percent / 100)
height = int(top.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
top = cv2.resize(top, dim, interpolation = cv2.INTER_AREA)

# convert to gray scale
topGray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
bottomGray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
for img, ax in zip([top,full,bottom], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('Image split')
axs[0].set_title('Top part')
axs[1].set_title('Full image')
axs[2].set_title('Bottom part')
plt.show()

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

# Uing Brute Force matcher with Hamming distance
# create BFMatcher object
distance_method = cv2.NORM_HAMMING
bf = cv2.BFMatcher(distance_method, crossCheck=True)

# Match descriptors
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.

#best_x_matches = len(matches)
best_x_matches = 150
img3 = cv2.drawMatches(topGray, kp1, bottomGray, kp2, matches[:best_x_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

f, ax = plt.subplots(figsize=(15,5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(img3)
ax.set_title('Matched ORB keypoints (best ' + str(best_x_matches)+ ' using Hamming distance)')
plt.show()

# %% Homography estimation and image stitching
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

output = cvfunctions.warpImages(bottom, top, H)

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
for img, ax in zip([bottom,output,top], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('Stitched image based on ORB matches')
axs[0].set_title('Bottom')
axs[1].set_title('Stitched image')
axs[2].set_title('Top')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15,6))
axs = axs.flatten()
for img, ax in zip([output,full], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('Stitched image based on ORB matches')
axs[0].set_title('Stitched image')
axs[1].set_title('Original')
plt.show()
# %%
