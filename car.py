#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.piCamera as piCamera
import lib.controller as controller

# Keyboard imput from terminal suffers from limitations on Linux due to
# permissions and udev. Using pygame as a non blocking method while not
# requiring root, May as well make use of pygame.camera to avoid
# any additional libraries being imported

import pygame
import pygame.camera

# arguement parsing
import argparse

# setup pin factory for all devices, default to pigpio to minimize stutter from software PWM
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, DistanceSensor

Device.pin_factory = PiGPIOFactory()

piCamera = piCamera.PiCamera()

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


carParser.add_argument('-d', '--data', metavar='data', type=bool,
                       nargs=1, default=False,
                       choices=[True, False],
                       help='Set to true for collecting training data')


args = carParser.parse_args()


if args.controller == 'manual':
    controller.keyboard(True, rcCar, servoLeftRight, servoUpDown, rcDistance, piCamera, args.data)
else:
    controller.ai()
