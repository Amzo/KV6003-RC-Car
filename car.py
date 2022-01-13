#!/usr/bin/env python

import lib.carSetup as carSetup
import lib.servoSetup as servo

# pigpio for PWM to prevent servo jitters in software gpio
import pigpio

# Keyboard imput from terminal suffers from limitations on Linux due to
# permissions and udev. Using pygame as a non blockign method while not
# requiring root

##### Create a new class for handling all this (seperate it) ##################################
import pygame
import pygame.camera

pygame.init()
pygame.camera.init()

window = pygame.display.set_mode((640, 480))
cam_list = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cam_list[0],(640,480))
cam.start()

##################################################

# initialize the car
rcCar = carSetup.Car()
rcCar.pinReset()

# initilize the motors
rcServoMotor = pigpio.pi()
leftRight    = 500;
upDown       = 500;

# switch from a while true loop
while True:

	########## move to camera class: getImage, transformImage methods, etc
	image1 = cam.get_image()
	image1 = pygame.transform.scale(image1,(640,480))
	window.blit(image1,(0,0))
	pygame.display.update()
	###################################################################

	for event in pygame.event.get():
		if event.type == pygame.KEYUP:
			rcCar.pinReset()
		if event.type == pygame.KEYDOWN:
			print(event.key)
			if event.key == pygame.K_w:
				rcCar.moveForward()
			if event.key == pygame.K_s:
				rcCar.moveBackwards()
			if event.key == pygame.K_a:
				rcCar.turnLeft()
			if event.key == pygame.K_d:
				rcCar.turnRight()
			if event.key == pygame.K_LEFT:
				increment += 300
				servo.turnMotor(rcServoMotor, 12, leftRight)
			if event.key == pygame.K_RIGHT:
				increment -= 300
				servo.turnMotor(rcServoMotor, 12, leftRight)
			if event.key == pygame.K_UP:
				increment += 300
				servo.turnMotor(rcServoMotor, 5, upDown)
			if event.key == pygame.K_DOWN:
				increment -= 300
				servo.turnMotor(rcServoMotor, 5, upDown)
