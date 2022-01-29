# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import pygame
import lib.cameraModule as cameraModule
import lib.directory as dir
import lib.trainedModel as  models
import gpiozero
import time, os

def ai(loop, rcCar, rcDistance, carCamera):
	model = models.load_model('models', 'model.tflite')

	while loop:
		piCamera.update_window()
		piCamera.get_image()
		aiKey =  models.get_prediction(model, carCamera.imageFrame, rcDistance.distance * 100)

		if aiKey == "w":
			rcCar.move_forward()
		elif aiKey == "a":
			rcCar.turn_left()
		elif aiKey == "d":
			rcCar.turn_right()
		elif aiKey == "s":
			rcCar.move_backwards()
		else:
			rcCar.close()

		# Don't want it to predict in real time, slow it down a few seconds
		time.sleep(0.5)
		rcCar.close()

		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					rcCar.close()
					piCamera.close()
					loop = False

def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, carCamera, dataArgs):
	if not dir.dir_exists(dataArgs.output[0]):
		os.makedirs(dataArgs.output[0])

	if dir.is_empty(dataArgs.output[0]):
		imageNumber = 1
	else:
		imageNumber = (dir.get_image_num(dataArgs.output[0]) + 1)

	# dictionary of event.Key to their corresponding keys for easier logging of data to csv
	inputKey = {
		"119": 'w',
		"97": 'a',
		"115": 's',
		"100": 'd'
	}

	while loop:
		carCamera.update_window()
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				rcCar.release()
			if event.type == pygame.KEYDOWN:
 				# get distance before executing a movement
				distance = rcDistance.distance * 100
				# only capture on asdw keys

				if dataArgs.data[0] and str(event.key) in inputKey and distance > 0:
					carCamera.data_capture(inputKey[str(event.key)], imageNumber, distance,  dataArgs.output[0])
					imageNumber += 1

				if event.key == pygame.K_w:
					rcCar.move_forward()
				elif event.key == pygame.K_s:
					rcCar.move_backwards()
				elif event.key == pygame.K_a:
					rcCar.turn_left()
				elif event.key == pygame.K_d:
					rcCar.turn_right()
				elif event.key == pygame.K_LEFT:
					servoLeftRight.turn_motor(10)
				elif event.key == pygame.K_RIGHT:
					servoLeftRight.turn_motor(-10)
				elif event.key == pygame.K_UP:
					servoUpDown.turn_motor(-10)
				elif event.key == pygame.K_DOWN:
					servoUpDown.turn_motor(10)
				elif event.key == pygame.K_q:
					# reset everything and exit
					rcCar.release()
					carCamera.release()
					loop = False;
				else:
					pygame.event.clear()
