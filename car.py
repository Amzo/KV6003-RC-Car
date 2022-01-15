#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.piCamera as piCamera
import lib.controller as controller

# Keyboard imput from terminal suffers from limitations on Linux due to
# permissions and udev. Using pygame as a non blocking method while not
# requiring root

import pygame
import pygame.camera

# arguement parsing
import argparse

# setup pin factory for all devices
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, DistanceSensor

Device.pin_factory = PiGPIOFactory()

window, cam = piCamera.initialize()

##################################################

# initialize the car and servo
rcCar = carSetup.Car()
servoLeftRight = carSetup.Servo(12)
servoUpDown = carSetup.Servo(5)

# initialize distance setting
rcDistance = DistanceSensor(echo=4, trigger=27)


# Add the arguments
carParser = argparse.ArgumentParser()
carParser = argparse.ArgumentParser(description='Flexible control for RC car')
carParser.add_argument('-c', '--controller', metavar='controller', type=str, 
                       nargs=1, default='manual',
                       choices=['manual', 'a.i'],
                       help='Specify controller to use')


args = carParser.parse_args()

for arg in vars(args):
    if getattr(args, arg) == 'manual':
        controller.keyboard(True, rcCar, servoLeftRight, servoUpDown, rcDistance)
    else:
        controller.ai()
    
