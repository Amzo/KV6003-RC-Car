#!/usr/bin/env python3
import pygame, cv2, csv, numpy, os

class PiCamera():
	def __init__(self):
		pygame.init()
		pygame.camera.init('OpenCV')
		self.imageFrame = None
		self.resizeDims = (200, 100)

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

	def processSurfaceImage(self):
		self.imageFrame = self.cam.get_image()

		# convert to 3D array for numpy
		surface3D = pygame.surfarray.array3d(self.imageFrame)
		surface3D = numpy.swapaxes(surface3D, 0, 1)

		surface3D = cv2.cvtColor(surface3D, cv2.COLOR_RGB2BGR)

		self.imageFrame = surface3D

	def dataCapture(self, input, number, distance, directory):
		self.imageFrame = pygame.surface.Surface((640, 480),0,self.window)
		self.processSurfaceImage()

		# resize and convert to grey scale
		self.transformGreyScale()

		# our y label will be input
		csvData = ["image{}.jpg".format(number), distance, input]

		self.imageResize()

		if not os.path.exists(directory):
			os.makedirs(directory)

		saveFile = directory + 'image{}.jpg'.format(number)
		print("saving file to {}".format(saveFile))
		cv2.imwrite(saveFile, self.imageFrame)

		self.writeCSV(csvData)

	# resize and transform on capture to allow using high resolution for viewing cam stream in window still
	def imageResize(self):
		self.imageFrame = cv2.resize(self.imageFrame, self.resizeDims, interpolation = cv2.INTER_AREA)

	def transformGreyScale(self):
		self.imageFrame = cv2.cvtColor(self.imageFrame, cv2.COLOR_BGR2GRAY)
