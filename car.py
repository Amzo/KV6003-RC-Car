#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.cameraModule as cameraModule
import lib.controller as controller
import lib.network as network

# argument parsing
import argparse

# setup pin factory for all devices, default to pigpio to minimize stutter from software PWM
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, DistanceSensor

Device.pin_factory = PiGPIOFactory()

# configuation parsing
import configparser

config = configparser.ConfigParser()
config.read('config/config.ini')

##################################################

# initialize the car and servo
rcCar = carSetup.Car()
servoLeftRight = carSetup.Servo(12, -10)
servoUpDown = carSetup.Servo(5, 10)

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
	# return 1 for error on unix just good practice
	quit(1)
else:
	if args.controller[0] == 'manual':
		try:
			carCamera = cameraModule.carCamera()
		except:
			print("Issues initializing pi camera")
		finally:
			controller.keyboard(True, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, args)
	elif args.controller[0] == 'ai':
		# stream video remotely to reduce pi load
		streamConnection = network.Server(config['host']['serverName'], int(config['port']['Port']))
		print("Starting video streaming server")
		streamConnection.start()
		print("Running A.I")
		controller.ai(True, rcCar, rcDistance, streamConnection)
	else:
		carParser.print_help()
		quit(1)
