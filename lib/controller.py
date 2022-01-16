# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import pygame
import lib.distanceSetup as distanceSetup
import lib.piCamera as piCamera

def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, piCamera, dataCapture):
	imageNumber = 1;

	# directionary of event.Key to their corresponding trees for easier logging of data to csv
	inputKey = {
		"119": 'w',
		"97": 'a',
		"115": 's',
		"100": 'd'
	}

	while loop:
		piCamera.updateWindow()
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				rcCar.stop()
			if event.type == pygame.KEYDOWN:
 				# get distance before executing a movement
				distance = distanceSetup.getDistance(rcDistance)
				# only capture on asdw keys
				if dataCapture and str(event.key) in inputKey and distance > 0:
					piCamera.dataCapture(inputKey[str(event.key)], imageNumber, distance,  "Data/")
					imageNumber += 1
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
					servoUpDown.turnMotor(-10)
				elif event.key == pygame.K_DOWN:
					servoUpDown.turnMotor(10)
				elif event.key == pygame.K_q:
					# reset everything and exit
					rcCar.stop()
					loop = False;
