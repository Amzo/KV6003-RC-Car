#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.cameraModule as cameraModule
import lib.controller as controller

# Keyboard imput from terminal suffers from limitations on Linux due to
# permissions and udev. Using pygame as a non blocking method while not
# requiring root

import pygame

# argument parsing
import argparse

# setup pin factory for all devices, default to pigpio to minimize stutter from software PWM
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, DistanceSensor

Device.pin_factory = PiGPIOFactory()

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
                       choices=['manual', 'ai'],
                       help='Specify controller to use')


carParser.add_argument('-d', '--data', metavar='data', type=bool,
                       nargs=1, default=False,
                       choices=[True, False],
                       help='Set to true for collecting training data')

carParser.add_argument('-o', '--output', metavar='output', type=str,
                       nargs=1, default="/home/pi/Data/Train/",
                       help="Specify output directory for data collection")


args = carParser.parse_args()

try:
    args.controller[0]
except NameError:
    carParser.print_help()
    quit(0)
else:
   # initialize the camera and run in it's own thread
    carCamera = cameraModule.carCamera()
    carCamera.run()

if args.controller[0] == 'manual' and carCamera is not None:
    controller.keyboard(True, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, args)
elif args.controller[0] == 'ai' and carCamera is not None:
    controller.ai(True, rcCar, rcDistance, carCamera)
else:
    carParser.print_help()
    quit(0)
