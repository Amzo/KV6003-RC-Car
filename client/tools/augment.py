#!/usr/bin/env python

# add a some augmentation to random images to balance out dataset
import random

import cv2
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

import os

import numpy as np

Tk().withdraw()
directory = askdirectory()

for filename in os.listdir(directory):
    i = os.path.join(directory, filename)
    img = cv2.imread(i)
    kernel = np.ones((5, 5), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # image = cv2.GaussianBlur(hsv, (3, 3), sigmaX=0, sigmaY=0)

    Lower_hsv = np.array([120, 120, 120])
    Upper_hsv = np.array([160, 170, 180])

    # sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=3)
    Mask = cv2.inRange(img, Lower_hsv, Upper_hsv)

    mask_yellow = cv2.bitwise_not(Mask)
    cv2.imshow('Mask', Mask)
    cv2.waitKey(0)
    Mask = cv2.bitwise_and(img, img, mask=mask_yellow)

    cv2.imshow('Mask', Mask)

    # waits for user to press any key
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    if not os.path.exists(f'{directory}/sobel'):
        os.mkdir(f'{directory}/sobel')

    # cv2.imwrite(f'{directory}/sobel/{filename}', sobelxy)
