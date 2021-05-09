import numpy as np
from cv2 import cv2

def apply_lowpass(img, radius):
    row, column = img.shape
    central_row, central_col = int(row / 2), int(column / 2)
    mask = np.zeros((row, column), np.uint8)
    img_center = (central_row, central_col)
    mask = cv2.circle(mask, img_center, radius, (1, 1, 1), -1)
    fshift = img * mask
    
    return mask, fshift

def apply_highpass(img, radius):
    row, column = img.shape
    central_row, central_col = int(row / 2), int(column / 2)
    mask = np.full((row, column), 1, np.uint8)
    img_center = (central_row, central_col)
    mask = cv2.circle(mask, img_center, radius, (0, 0, 0), -1)
    
    fshift = img * mask
    
    return mask, fshift

def apply_bandpass(img, radius_low, radius_high):
    lowpass, _ = apply_lowpass(img, radius_low)
    highpass, _ = apply_highpass(img, radius_high)
    
    mask = lowpass + highpass
    mask[mask < 2] = 0
    
    fshift = img * mask
    
    return mask, fshift