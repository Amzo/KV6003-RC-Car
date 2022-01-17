# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import pygame
import lib.piCamera as piCamera
import lib.directory as dir
import lib.trainedModel as  models

def ai(loop, rcCar, rcDistance, piCamera):
	model = models.load_model('models', 'model.tflite')

	while loop:
		piCamera.update_window()
		piCamera.get_image()
		aiKey =  models.get_prediction(model, piCamera.imageFrame, rcDistance.distance * 100)

		if ai.key == "w":
			rcCar.move_forward()
		elif ai.key == "a":
			rcCar.turn_left()
		elif ai.key == "d":
			rcCar.turn_right()
		else:
			rcCAr.move_backwards()

def keyboard(loop, rcCar, servoLeftRight, servoUpDown, rcDistance, piCamera, dataCapture):
	if dir.is_empty("Data/"):
		imageNumber = 1
	else:
		imageNumber = (dir.get_image_num("Data/") + 1)
		print(imageNumber)
	# directionary of event.Key to their corresponding trees for easier logging of data to csv
	inputKey = {
		"119": 'w',
		"97": 'a',
		"115": 's',
		"100": 'd'
	}

	while loop:
		piCamera.update_window()
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				rcCar.close()
			if event.type == pygame.KEYDOWN:
 				# get distance before executing a movement
				distance = rcDistance.distance * 100
				# only capture on asdw keys
				if dataCapture and str(event.key) in inputKey and distance > 0:
					piCamera.data_capture(inputKey[str(event.key)], imageNumber, distance,  "Data/")
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
					rcCar.close()
					piCamera.close()
					loop = False;
