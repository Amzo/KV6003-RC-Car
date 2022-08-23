#!/usr/bin/env python

# add a some augmentation to random images to balance out dataset
import random

import cv2
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

import os

Tk().withdraw()
filename = askdirectory()


def getFileCount(filename):
    files = next(os.walk(filename))[2]
    return len(files)


# modified from:
# https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
def add_noise(img):
    # Getting the dimensions of the image
    row, col, ch = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 500)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


# make them up to 12,000 images each
count = 3000 - getFileCount(filename)
print("Done")

for x in range(1, count + 1):
    image = None
    saveFile = None
    while image is None:
        y = random.randint(24, 7210)
        imageFile = f'{filename}/image{y}.jpg'
        if os.path.exists(imageFile):
            image = cv2.imread(imageFile)

    while saveFile is None:
        fileNum = random.randint(12000, 2400000)

        if not os.path.exists(f'{filename}/image{fileNum}.jpg'):
            saveFile = f'{filename}/image{fileNum}.jpg'
            imagenew = add_noise(image)
            cv2.imwrite(saveFile, imagenew)

    print(x)
