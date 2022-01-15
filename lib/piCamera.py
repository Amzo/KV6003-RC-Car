#!/usr/bin/env python3
import pygame
import cv2
import csv

def initialize():
	pygame.init()
	pygame.camera.init()

	window = pygame.display.set_mode((640, 480))
	cam_list = pygame.camera.list_cameras()
	cam = pygame.camera.Camera(cam_list[0],(640,480))
	cam.start()

	return window, cam


def updateWindow(cam, window):
	frame = cam.get_image()
	frame = pygame.transform.scale(frame,(640,480))
	window.blit(frame,(0,0))
	pygame.display.update()

def writeCSV(data):
	with open("labels.csv", 'a', encoding='UTF8') as f:
		writer = csv.writer(f)
		writer.writerow(data)

def dataCapture(cam, input, number):
	csvData = ["image{}.jpg".format(number), input]
	frame = cam.get_image()
	data = pygame.surfarray.array2d(frame)
	cv2.imwrite("~/Data/image{}.jpg".format(number), data)
	writeCSV(csvData)
