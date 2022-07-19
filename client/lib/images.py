import os

import cv2
import numpy as np


def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gauss(image):
    return cv2.GaussianBlur(image, (9, 9), 0)


def canny(image):
    edges = cv2.Canny(image, 75, 30)
    return edges


def region(image):
    height, width = image.shape
    triangle = np.array([
        [(100, height), (320, 240), (width, height)]
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


#for x in os.listdir("/home/amzo/University/Year3/Data/Train/w/"):
#    image = cv2.imread(f"/home/amzo/University/Year3/Data/Train/w/{x}")

def linedImage(image):
    img = grey(image)
    img = gauss(img)
    img = canny(img)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, None, 30, 50)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,0), 3, cv2.LINE_AA)
        # image = region(image)
        return image
    else:
        return None

