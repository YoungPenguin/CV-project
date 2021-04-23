## ORB

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