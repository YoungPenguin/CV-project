#!/usr/bin/env python3

### Harris corners

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

# find corners
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