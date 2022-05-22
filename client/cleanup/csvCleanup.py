import csv
import os
import sys
import time
from _csv import reader
from typing import Any

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
           "X", "Y"]

data = csv.reader(open('C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\data.csv'))

for i in letters:
    directory = 'C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\Test\\'
    for file in os.listdir(directory + '{}\\'.format(i)):
        with open('C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\data.csv', 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                if row[0] == file and row[-1] == i:
                    with open('{}/newDdata.csv'.format('C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\'), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)








