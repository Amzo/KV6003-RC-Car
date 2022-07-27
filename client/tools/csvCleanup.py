import os
from _csv import reader
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

import cv2

Tk().withdraw()
filename = askdirectory()

try:
    with open(f"{filename}/labels.csv", 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            print(row[0])
            image = cv2.imread(f"{filename}/{row[0]}")
            cv2.imshow(f"Image Check: {row[0]}", image)
            k = cv2.waitKey()
            if k == 100:
                os.remove(f"{filename}/{row[0]}")

            # open file to check
            cv2.destroyAllWindows()
except (FileNotFoundError, AttributeError):
    pass
