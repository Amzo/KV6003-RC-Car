#!/usr/bin/env python3
import pygame
import cv2
import csv

class PiCamera():
	def __init__(self):
		pygame.init()
		pygame.camera.init()

		self.window = pygame.display.set_mode((640, 480))
		camList = pygame.camera.list_cameras()
		self.cam = pygame.camera.Camera(camList[0],(640,480))
		self.cam.start()


	def updateWindow(self):
		frame = self.cam.get_image()
		frame = pygame.transform.scale(frame,(640,480))
		self.window.blit(frame,(0,0))
		pygame.display.update()

	def writeCSV(self, data):
		with open("labels.csv", 'a', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(data)

	def dataCapture(self, input, number):
		csvData = ["image{}.jpg".format(number), input]
		frame = self.cam.get_image()
		data = pygame.surfarray.array2d(frame)
		cv2.imwrite("~/Data/image{}.jpg".format(number), data)
		writeCSV(csvData)
