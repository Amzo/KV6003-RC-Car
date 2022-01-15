#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.distanceSetup as distanceSetup
import lib.piCamera as piCamera

# Keyboard imput from terminal suffers from limitations on Linux due to
# permissions and udev. Using pygame as a non blocking method while not
# requiring root

import pygame
import pygame.camera

# arguement parsing
import argparse

# setup pin factory for all devices
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo, DistanceSensor

Device.pin_factory = PiGPIOFactory()

window, cam = piCamera.initialize()

##################################################

# initialize the car and servo
rcCar = carSetup.Car()
servoLeftRight = carSetup.Servo(12)
servoUpDown = carSetup.Servo(5)

#servoLeftRight = AngularServo(12, min_angle=-90, max_angle=90)

# initialize distance setting
rcDistance = DistanceSensor(echo=4, trigger=27)

# Arguements
parser = argparse.ArgumentParser(description='RC car startup options')
parser.add_argument('-c','--capture', help='Capture keyboard input and frames for training', required=False)
parser.add_argument('-a','--ai', help='Let the AI control the car', required=True)

# switch from a while true loop
while True:

	########## move to camera class: getImage, transformImage methods, etc
	piCamera.updateWindow(cam, window)
	###################################################################

	for event in pygame.event.get():
		if event.type == pygame.KEYUP:
			rcCar.stop()
		if event.type == pygame.KEYDOWN:
			# get distance before executing a movement
			distanceSetup.getDistance(rcDistance)
			if event.key == pygame.K_w:
				rcCar.moveForward()
			elif event.key == pygame.K_s:
				rcCar.moveBackwards()
			elif event.key == pygame.K_a:
				rcCar.turnLeft()
			elif event.key == pygame.K_d:
				rcCar.turnRight()
			elif event.key == pygame.K_LEFT:
                servoLeftRight.turnMotor(10)
			elif event.key == pygame.K_RIGHT:
				servoLeftRight.turnMotor(-10)
			elif event.key == pygame.K_UP:
				servoUpDown(-10)
			elif event.key == pygame.K_DOWN:
				servoUpDown(10)
			elif event.key == pygame.K_q:
				# reset everything and exit
				rcCar.stop()
				#servo.turnMotor(rcServorMotor,5, 0)
