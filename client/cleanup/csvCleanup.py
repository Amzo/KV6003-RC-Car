import csv
import os
import shutil
import sys
import time
from _csv import reader
import re
import cv2 as cv
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

Tk().withdraw()
filename = askdirectory()

try:
    with open(f"{filename}/labels.csv", 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            print(row[0])
            image = cv.imread(f"{filename}/{row[0]}")
            cv.imshow(f"Image Check: {row[0]}", image)
            k = cv.waitKey()
            if k == 100:
                os.remove(f"{filename}/{row[0]}")

            # open file to check
            cv.destroyAllWindows()
except (FileNotFoundError, AttributeError):
    pass
