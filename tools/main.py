#!/usr/bin/env python3

import argparse
from os.path import exists, isdir
from os import makedirs
from csv import reader, writer
import shutil


# Tool used for moving old data collection type to the new format
# where each image is saved in it's class directory, to help find class
# imbalances
def move_to_label_dir(image, path, dirs, output):
    save_location = output + '\\'

    if not exists(save_location + path):
        makedirs(save_location + path)
    print(dirs + image)
    shutil.copyfile(dirs + '\\' + image, save_location + path + "\\" + image)


def read_csv_file(filename, output):
    with open(filename + '\\labels.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            move_to_label_dir(row[0], row[2], filename, output)
            row[0] = row[2] + "/" + row[0]
            with open(output + '\\labels.csv', 'a', newline='') as n:
                writeFile = writer(n)
                writeFile.writerow(row)


def parse_csv_file(filename, output):
    if exists(filename):
        read_csv_file(filename, output)
    else:
        print("Couldn't find file {}".format(filename))


def dir_path(path):
    if isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Directory: {path} doesn't exist")


if __name__ == '__main__':
    dirParser = argparse.ArgumentParser(description='input and output directory')

    dirParser.add_argument('-i', '--input', metavar='inputDir', type=dir_path,
                           nargs=1, required=True,
                           help='input directory of csv file and training images')

    dirParser.add_argument('-o', '--output', metavar='output', type=dir_path,
                           nargs=1, required=True,
                           help="specify output of cleaned data")

    cmdArgs = dirParser.parse_args()

    try:
        cmdArgs.input[0]
    except (NameError, AttributeError):
        dirParser.print_help()
        # return 1 for error on unix just good practice
        quit(1)
    else:
        parse_csv_file(cmdArgs.input[0], cmdArgs.output[0])
