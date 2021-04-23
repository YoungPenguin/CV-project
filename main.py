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

fig, axs = plt.subplots(1, 3)
axs = axs.flatten()
for img, ax in zip([top,full,bottom], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('Image split')
axs[0].set_title('Top part')
axs[1].set_title('Full image')
axs[2].set_title('Bottom part')

#%% Feature extraction and description
cv2.destroyAllWindows()
# ORB
# create ORB dectector
orb = cv2.ORB_create(nfeatures=3000)  # default features is 500

# find key point and descriptor
kp1, des1 = orb.detectAndCompute(topGray, None)
kp2, des2 = orb.detectAndCompute(bottomGray, None)
print("Keypoints")
print(len(kp1))
print(len(kp2))

# draw key point on image
result_im1 = cv2.drawKeypoints(topGray, kp1, None, flags=0)
result_im2 = cv2.drawKeypoints(bottomGray, kp2, None, flags=0)

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
for img, ax in zip([result_im1,result_im2], axs):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img)
fig.suptitle('ORB keypoints')
axs[0].set_title('Top')
axs[1].set_title('Bottom part')
plt.show()
cv2.imshow("Top image",result_im1)
cv2.imshow("Bottom image",result_im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Uing Brute Force matcher with Hamming distance
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(topGray, kp1, bottomGray, kp2, matches[:150],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

f, ax = plt.subplots()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(img3)
ax.set_title('ORB detector and descriptor, BF matcher')
plt.show()

#cv2.imshow("ORB detector and descriptor, BF matcher", img3)

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
plt.title = 'Blobs'
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
cv2.imshow('SIFT', img3)

# Find homography

flattened = [val for sublist in good for val in sublist]

src_pts = np.float32([ kp1[m.queryIdx].pt for m in flattened ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in flattened ]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
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
img3 = cv2.drawMatches(topGray,kp1,bottomGrayLine,kp2,flattened,None,**draw_params)
cv2.imshow('gray', img3)
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

#%% Harris corners (brief descriptors)
cv2.destroyAllWindows()
def brief_descriptor(im1, im2, cim1, cim2):

    extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kpts1_prep = list(zip(np.array(cim1[0]).astype(float), np.array(cim1[1]).astype(float)))
    keypoints_im1 = [cv2.KeyPoint(x[1], x[0], 1) for x in kpts1_prep]

    kpts2_prep = list(zip(np.array(cim2[0]).astype(float), np.array(cim2[1]).astype(float)))
    keypoints_im2 = [cv2.KeyPoint(x[1], x[0], 1) for x in kpts2_prep]

    (kps1, features1) = extractor.compute(im1, keypoints_im1)
    (kps2, features2) = extractor.compute(im2, keypoints_im2)
    return kps1, features1, kps2, features2

kps1, f1, kps2, f2 = brief_descriptor(topGray, bottomGray, ctop, cbot)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(f1, f2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(topGray, kps1, bottomGray, kps2, matches[:150],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Harris corners and brief descriptor, BF matcher",img3)

#%%
descriptTop, pts_top = cvfunctions.simpleDescriptor(topGray, ctop, 7)
descriptBot, pts_bot = cvfunctions.simpleDescriptor(bottomGray, cbot, 7)

matches = bf.knnMatch(f1,f2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))

plt.figure(figsize=(30,20))
img3 = cv2.drawMatchesKnn(topGray,kps1,bottomGray,kps2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

# Using Brute Force matcher with Hamming distance
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # to use L2-norm, write cv2.NORM_L2

# Match descriptors
#matches = bf.match(f1,f2)
matches = bf.match(descriptTop,descriptBot)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
#img3 = cv2.drawMatches(topGray, kps1, bottomGray, kps2, matches[:150],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.drawMatches(topGray, pts_top, bottomGray, pts_bot, matches[:150],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Harris corner detector and BRIEF descriptor, BF matcher",img3)
#%%
## Detect, descript, match, stitch using Homography matrix


## Feature matching
# Use Hamming distance for ORB

## Homography estimation (week 5)
# Using SVD

## Image stitching
# Using homography matrix





