#!/usr/bin/env python
import lib.carSetup as carSetup
import lib.servoSetup as servoSetup
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
servoLeftRight = AngularServo(12, min_angle=-90, max_angle=90)
servoUpDown = AngularServo(5, min_angle=-90, max_angle=90)

# initialize distance setting
rcDistance = DistanceSensor(echo=4, trigger=27)

# Angular position starting point. Range -90 to +90
leftRight    = 0;
upDown       = 0;

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
				leftRight += 10
				servoSetup.turnMotor(servoLeftRight, leftRight)
			elif event.key == pygame.K_RIGHT:
				leftRight -= 10
				servoSetup.turnMotor(servoLeftRight, leftRight)
			elif event.key == pygame.K_UP:
				upDown -= 10
				servoSetup.turnMotor(servoUpDown, upDown)
			elif event.key == pygame.K_DOWN:
				upDown += 10
				servoSetup.turnMotor(servoUpDown, upDown)
			elif event.key == pygame.K_q:
				# reset everything and exit
				rcCar.stop()
				#servo.turnMotor(rcServorMotor,5, 0)
