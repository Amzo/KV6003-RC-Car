# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 08:35:01 2022

@author: Anthony Donnelly
"""
import pygame
import lib.directory as dir
import time, os

def ai(loop, rcCar, rcDistance, streamConnection):
	while loop:
		print("Getting command")
		streamConnection.getCommand()
		# ignore empty strings in the data stream
		aiKey = ''.join(streamConnection.commands).split()
		try :
			aiKey = aiKey[0]
		except IndexError:
			# client side might still be processing
			pass
			
		print("prediction key is: ", aiKey)
		if aiKey == "w":
			print("moving forward")
			rcCar.move_forward()
		elif aiKey == "a":
			rcCar.turn_left()
		elif aiKey == "d":
			rcCar.turn_right()
		elif aiKey == "s":
			rcCar.move_backwards()
		elif aiKey == "q":
			rcCar.turn_left_90()
		elif aiKey == "e":
			rcCar.turn_right_90()
		else:
			rcCar.release()
			
		time.sleep(0.1)
		rcCar.release()

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
		"100": 'd',
		"101": 'e',
		"113": 'q',
		"116": 't'
	}

	while loop:
		carCamera.update_window()
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				rcCar.release()
			if event.type == pygame.KEYDOWN:
 				# get distance before executing a movement
				distance = rcDistance.distance * 100
				#print(distance)
				# only capture on asdw keys

				if dataArgs.data[0] and str(event.key) in inputKey and distance > 0:
					carCamera.data_capture(inputKey[str(event.key)], imageNumber, distance,  dataArgs.output[0])
					print("Captured image number {}".format(imageNumber))
					imageNumber += 1
				if event.key == pygame.K_w:
					rcCar.move_forward()
				elif event.key == pygame.K_s:
					rcCar.move_backwards()
				elif event.key == pygame.K_a:
					rcCar.turn_left()
				elif event.key == pygame.K_d:
					rcCar.turn_right()
				elif event.key == pygame.K_e:
					rcCar.turn_right_90()
				elif event.key == pygame.K_q:
					rcCar.turn_left_90()
				elif event.key == pygame.K_t:
					print(event.key)
					rcCar.release()
				elif event.key == pygame.K_LEFT:
					servoLeftRight.turn_motor(10)
				elif event.key == pygame.K_RIGHT:
					servoLeftRight.turn_motor(-10)
				elif event.key == pygame.K_UP:
					servoUpDown.turn_motor(-10)
				elif event.key == pygame.K_DOWN:
					servoUpDown.turn_motor(10)
				elif event.key == pygame.K_z:
					# reset everything and exit
					rcCar.release()
					carCamera.release()
					loop = False;
				else:
					pygame.event.clear()
