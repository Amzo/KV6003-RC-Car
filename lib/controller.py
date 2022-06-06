# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import os
import time

import pygame

import lib.directory as myDir


def ai(loop, rcCar, servoUpDown, servoLeftRight, rcDistance, streamConnection):
    while loop:
        print("Getting command")
        streamConnection.getCommand()
        # ignore empty strings in the data stream
        aiKey = ''.join(streamConnection.commands).split()
        try:
            aiKey = aiKey[0]
        except IndexError:
            # client side might still be processing
            pass

        if aiKey in ["l", "c", "t", "r"] and rcDistance.distance < 30:
            if aiKey == "r":
                rcCar.turn_right_90()
            elif aiKey == "l":
                rcCar.turn_left_90()
            else:
                rcCar.release()
        else:
            if aiKey == "w":
                print("moving forward")
                rcCar.move_forward()
            elif aiKey == "a":
                rcCar.turn_left()
            elif aiKey == "d":
                rcCar.turn_right()
            elif aiKey == "s":
                rcCar.move_backwards()
            elif aiKey == "q":
                rcCar.turn_left_90()
            elif aiKey == "e":
                rcCar.turn_right_90()
            elif aiKey == "i":
                servoUpDown.turn_motor(-10)
            elif aiKey == "k":
                servoUpDown.turn_motor(+10)
            elif aiKey == "j":
                servoLeftRight.turn_motor(10)
            elif aiKey == "l":
                servoLeftRight.turn_motor(-10)
            else:
                rcCar.release()

        time.sleep(0.1)
        rcCar.release()


def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, dataArgs):
    if not myDir.dir_exists(dataArgs.output[0]):
        os.makedirs(dataArgs.output[0])

    if myDir.is_empty(dataArgs.output[0]):
        imageNumber = 1
    else:
        imageNumber = (myDir.get_image_num(dataArgs.output[0]) + 1)

#    carCamera.imageNumber = imageNumber
#    carCamera.start_capture()
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

    prevKey = 0
    while loop:
        carCamera.update_window()
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                rcCar.release()
                carCamera.capture = False
            if event.type == pygame.KEYDOWN:
                # get distance before executing a movement
                distance = rcDistance.distance * 100
                # print(distance)
                # only capture on asdw keys

                if inputKey[str(event.key)] not in dataArgs.ignore:
                    if dataArgs.data[0] and str(event.key) in inputKey and distance > 0:
                        carCamera.data_capture(inputKey[str(event.key)], imageNumber, distance, prevKey,
                                               dataArgs.output[0])
                        print("Captured image number {}".format(imageNumber))
                        imageNumber += 1
                else:
                    print("ignorign capture of key {}".format(inputKey[str(event.key)]))
                if event.key == pygame.K_w:
                    rcCar.move_forward()
                elif event.key == pygame.K_s:
                    rcCar.move_backwards()
                elif event.key == pygame.K_a:
                    rcCar.turn_left()
                elif event.key == pygame.K_d:
                    rcCar.turn_right()
                elif event.key == pygame.K_e:
                    rcCar.turn_right_90()
                elif event.key == pygame.K_q:
                    rcCar.turn_left_90()
                elif event.key == pygame.K_t:
                    print(event.key)
                    rcCar.release()
                elif event.key == pygame.K_LEFT:
                    servoLeftRight.turn_motor(10)
                elif event.key == pygame.K_RIGHT:
                    servoLeftRight.turn_motor(-10)
                elif event.key == pygame.K_UP:
                    servoUpDown.turn_motor(-10)
                elif event.key == pygame.K_DOWN:
                    servoUpDown.turn_motor(10)
                elif event.key == pygame.K_z:
                    # reset everything and exit
                    rcCar.release()
                    carCamera.release()
                    loop = False
                else:
                    pygame.event.clear()

                if inputKey[str(event.key)] == "w":
                    prevKey = 0.33
                elif inputKey[str(event.key)] == "a":
                    prevKey = 0.66
                elif inputKey[str(event.key)] == "d":
                    prevKey = 0.99
