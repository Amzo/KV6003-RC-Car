#!/usr/bin/env python3
import pygame, cv2, csv, numpy, os
import lib.directory as dir

class PiCamera():
	def __init__(self):
		pygame.init()
		pygame.camera.init('OpenCV')
		self.imageFrame = None

		# Confused me for awhile numpy uses (height, width) while CV2 uses (width, height)
		self.resizeDims = (100, 200)

		self.window = pygame.display.set_mode((640, 480))
		camList = pygame.camera.list_cameras()
		self.cam = pygame.camera.Camera(camList[0],(640,480))
		self.cam.start()


	def update_window(self):
		frame = self.cam.get_image()
		frame = pygame.transform.scale(frame,(640,480))
		self.window.blit(frame,(0,0))
		pygame.display.update()

	def write_csv(self, data):
		with open("Data/labels.csv", 'a', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(data)

	def process_surface_image(self):
		self.imageFrame = self.cam.get_image()

		# convert to 3D array for numpy
		surface3D = pygame.surfarray.array3d(self.imageFrame)
		#surface3D = numpy.transpose(surface3D, 0, 1)

#		print(surface3D.shape)
#		surface3D = cv2.cvtColor(surface3D, cv2.COLOR_RGB2BGR)

#		print(surface3D.shape)
		self.imageFrame = surface3D

	def get_image(self):
		self.imageFrame = pygame.surface.Surface((640, 480),0,self.window)
		self.process_surface_image()

                # resize and convert to grey scale
		self.transform_grey_scale()

		self.image_resize()

	def data_capture(self, input, number, distance, directory):
		self.imageFrame = pygame.surface.Surface((640, 480),0,self.window)
		self.process_surface_image()

		# resize and convert to grey scale
		self.transform_grey_scale()

		# our y label will be input
		csvData = ["image{}.jpg".format(number), distance, input]

		self.image_resize()

		if not dir.dir_exists(directory):
			os.makedirs(directory)

		saveFile = directory + 'image{}.jpg'.format(number)
		print("saving file to {}".format(saveFile))
		cv2.imwrite(saveFile, self.imageFrame)

		self.write_csv(csvData)

	# resize and transform on capture to allow using high resolution for viewing cam stream in window still
	def image_resize(self):
		self.imageFrame = cv2.resize(self.imageFrame, self.resizeDims)

	def transform_grey_scale(self):
		self.imageFrame = cv2.cvtColor(self.imageFrame, cv2.COLOR_RGB2GRAY)

	def close(self):
		self.cam.stop()
		pygame.quit()
