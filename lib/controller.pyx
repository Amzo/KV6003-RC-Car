# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import csv
import os
import sys
import time

import pygame

import lib.directory as myDir


def ai(loop, rcCar, servoUpDown, servoLeftRight, rcDistance, rcDistance2, streamConnection):
    prevKey = []
    while loop:
        print("Getting command")
        if len(prevKey) >= 5:
            prevKey = []
        streamConnection.getCommand()
        # ignore empty strings in the data stream
        aiKey = ''.join(streamConnection.commands).split()
        print(f'Distance sensor1 is reading {rcDistance.distance}')
        print(f'Distance sensor2 is reading {rcDistance2.distance}')

        try:
            aiKey = aiKey[0]
            pred = list(aiKey)
            print(pred)
        except IndexError:
            # client side might still be processing
            pass

        if rcDistance.distance < 0.30 or rcDistance2.distance < 0.30:
            print(f'Got: {pred[0]}')
            if pred[0] == "r":
                prevKey.append(1)
                if len(prevKey) == 5 and max(prevKey,key=prevKey.count) == 1:
                    rcCar.turn_right_90()
            elif pred[0] == "l":
                prevKey.append(0)
                if len(prevKey) == 5 and max(prevKey, key=prevKey.count) == 0:
                    rcCar.turn_left_90()
            elif pred[0] == "t":
                print(f'Stop sign at a distance of {rcDistance2.distance * 100}cm')
                rcCar.release()
            elif pred[0] == "p":
                print(f'Pedestrian at a Distance of {rcDistance2.distance * 100}cm')
                rcCar.release()
            elif pred[0] == "c":
                print(f'Car at a Distance is {rcDistance2.distance * 100}')
                rcCar.release()
        else:
            if pred[1] == "w":
                print("moving forward")
                rcCar.move_forward()
            elif pred[1] == "a":
                rcCar.turn_left()
            elif pred[1] == "d":
                rcCar.turn_right()
            elif pred[1] == "s":
                rcCar.move_backwards()
            elif pred[1] == "q":
                rcCar.turn_left_90()
            elif pred[1] == "e":
                rcCar.turn_right_90()
            elif pred[1] == "i":
                servoUpDown.turn_motor(-10)
            elif pred[1] == "k":
                servoUpDown.turn_motor(+10)
            elif pred[1] == "j":
                servoLeftRight.turn_motor(10)
            elif pred[1] == "l":
                servoLeftRight.turn_motor(-10)
            else:
                rcCar.release()

        time.sleep(0.1)
        rcCar.release()


def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, dataArgs):
    if not myDir.dir_exists(dataArgs.output[0]):
        os.makedirs(dataArgs.output[0])

    if myDir.is_empty(dataArgs.output[0]):
        carCamera.imageNumber = 1
    else:
        carCamera.imageNumber = (myDir.get_image_num(dataArgs.output[0]) + 1)

    # dictionary of event.Key to their corresponding keys for easier logging of data to csv
    inputKey = {
        "119": 'w',
        "97": 'a',
        "115": 's',
        "100": 'd',
        "101": 'e',
        "113": 'q',
        "116": 't'
    }

    # pass the data save location to the camera class to avoid it being done repeatedly in the loop
    carCamera.saveDirectory = dataArgs.output[0]

    def capture(key):
        if inputKey[str(key)] not in dataArgs.ignore:
            carCamera.key = inputKey[str(key)]
            carCamera.data_capture()
            carCamera.imageNumber += 1

    # more efficient to open file and keep it open
    if not os.path.exists(carCamera.saveDirectory + "labels.csv"):
        open(carCamera.saveDirectory + "labels.csv", 'a').close()
    carCamera.csvFile = open(carCamera.saveDirectory + "labels.csv", 'a')
    carCamera.writer = csv.writer(carCamera.csvFile)

    while loop:
        #CarCamera.update_window()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            capture(pygame.K_w)
            rcCar.move_forward()
            carCamera.prevKey = 0.10
        elif keys[pygame.K_s]:
            capture(pygame.K_s)
            rcCar.move_backwards()
            carCamera.prevKey = 0.20
        elif keys[pygame.K_a]:
            capture(pygame.K_a)
            rcCar.turn_left()
            carCamera.prevKey = 0.30
        elif keys[pygame.K_d]:
            capture(pygame.K_d)
            rcCar.turn_right()
            carCamera.prevKey = 0.40
        elif keys[pygame.K_e]:
            rcCar.turn_right_90()
            carCamera.prevKey = 0.50
        elif keys[pygame.K_q]:
            capture(pygame.K_q)
            #rcCar.turn_left_90()
            carCamera.prevKey = 0.60
        elif keys[pygame.K_t]:
             capture(pygame.K_t)
             rcCar.release()
             carCamera.prevKey = 0.70
        elif keys[pygame.K_z]:
            # reset everything and exit
            rcCar.release()
            carCamera.release()
            loop = False
            carCamera.csvFile.close()
        else:
            rcCar.release()
            pygame.event.clear()
        try:
            for e in pygame.event.get():
                pass
        except pygame.error:
            # probably closed
            sys.exit(0)

