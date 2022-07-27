#!/usr/bin/env python
# argument parsing
import argparse
# configuration parsing
import configparser

from gpiozero import Device, DistanceSensor
# setup pin factory for all devices, default to pigpio to minimize stutter from software PWM
from gpiozero.pins.pigpio import PiGPIOFactory

import lib.cameraModule as cameraModule
import lib.carSetup as carSetup
import lib.controller as controller
import lib.network as network

# profiling to find bottleneck

Device.pin_factory = PiGPIOFactory()

config = configparser.ConfigParser()
config.read('config/config.ini')

##################################################

# initialize the car and servo
rcCar = carSetup.Car()
servoLeftRight = carSetup.Servo(12, -10)
servoUpDown = carSetup.Servo(5, 10)

# initialize distance setting
rcDistance = DistanceSensor(echo=4, trigger=27)
rcDistance2 = DistanceSensor(echo=22, trigger=17)

# Add the arguments
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

carParser.add_argument('-i', '--ignore', metavar='ignore', type=str,
                       nargs=3, default="l",
                       help="Ignore key in data collection")

args = carParser.parse_args()

if __name__ == '__main__':
    try:
        args.controller[0]
    except NameError:
        carParser.print_help()
        # return 1 for error on unix just good practice
        quit(1)
    else:
        if args.controller[0] == 'manual':
            try:
                carCamera = cameraModule.CarCamera()
            finally:
                controller.keyboard(True, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, args)
        elif args.controller[0] == 'ai':
            streamConnection = network.start_connection()
            controller.ai(True, rcCar, servoUpDown, servoLeftRight, rcDistance, rcDistance2, streamConnection)
        else:
            carParser.print_help()
            quit(1)
