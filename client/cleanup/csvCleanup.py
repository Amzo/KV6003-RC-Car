import csv
import os
import shutil
import sys
import time
from _csv import reader
import re

try:
    with open('C:\\Users\\Amzo\\Documents\\RC Car Data\\NewData\\Train\\' + "labels.csv", 'r') as file:
        data = file.readlines()

    lastImage = data[-1].split(',')[0]

    imageNumber = int(re.search(r'\d+', lastImage).group(0))

except (FileNotFoundError, AttributeError):
    pass

with open('C:\\Users\\Amzo\\Documents\\RC Car Data\\NewData\\Train1\\labels.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print("copying file " + row[0] + " to " + row[2])
        shutil.copy('C:\\Users\\Amzo\\Documents\\RC Car Data\\NewData\\Train1\\' + row[0],
                    'C:\\Users\\Amzo\\Documents\\RC Car Data\\NewData\\Train\\' + row[2] + "\\" + "image{}.jpg".format(imageNumber))

        row[0] = row[2] + "/" + "image{}.jpg".format(imageNumber)
        with open('{}/labels.csv'.format('C:\\Users\\Amzo\\Documents\\RC Car Data\\NewData\\Train\\'),
                  'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        imageNumber += 1
