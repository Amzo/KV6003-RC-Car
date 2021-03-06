#!/usr/bin/env python3
import os
import re


def is_empty(directory):
    if len(os.listdir(directory)) == 0:
        return True
    else:
        return False


def dir_exists(directory):
    if os.path.exists(directory):
        return True
    else:
        return False


# Return last file number. As all files are saved as 'imageXXX.jpg' glob and max is fine
def get_image_num(directory):
    try:
        with open(directory + "labels.csv", 'r') as file:
            data = file.readlines()

        lastImage = data[-1].split(',')[0]

        return int(re.search(r'\d+', lastImage).group(0))
    except (FileNotFoundError, AttributeError):
        return 0
