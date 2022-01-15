# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import pygame
import lib.distanceSetup as distanceSetup
import lib.piCamera as piCamera

def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, window, cam, dataCapture):
	imageNumber = 1;
	while loop:
		piCamera.updateWindow(cam, window)
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				rcCar.stop()
			if event.type == pygame.KEYDOWN:
 				# get distance before executing a movement
				distanceSetup.getDistance(rcDistance)
				if dataCapture:
					print("Saving image {}".format(imageNumber))
					piCamera.dataCapture(cam, event.key, imageNumber)
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
