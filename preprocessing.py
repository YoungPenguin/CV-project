import cv2
import numpy as np
import math

def splitY(overlap, image):
    assert 0 <= overlap and overlap < 1, "overlapping area should be a positive percentage less than 1 in decimal notation"
    print(image.shape)
    height = image.shape[1]
    top_end = math.ceil((height + height*overlap) / 2)
    bottom_start = height - top_end
    top = image[:top_end]
    bottom = image[bottom_start:]
    return image,top,bottom

def splitX(overlap, image):
    assert 0 < overlap and overlap < 1, "overlap should be a decimal between 0 and 1"
    print(image.shape)
    width = image.shape[0]
    left_end = math.ceil((width + width*overlap) / 2)
    right_start = width - left_end
    left = image[:,:left_end]
    right = image[:,right_start:]
    return image,left,right