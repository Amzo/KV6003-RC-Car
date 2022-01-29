##!/usr/bin/env python3
import pygame, cv2, csv
import picamera, io
import threading

class carCamera():
	def __init__(self):
		pygame.init()
		self.camera = picamera.PiCamera()
		self.camera.resolution = (640, 480)

		self.imageFrame = None
		self.resizeDims = (200, 100)
		self.window = pygame.display.set_mode((640, 480))
		self.loopFlag = True

		self.cameraThread = None

	def start(self):
		self.cameraThread = threading.Thread(target=self.update_window, args=(), daemon=True)
		self.cameraThread.start()
		#pool = ThreadPool(processes=1)
		#pool.apply_async(self.update_window, args=())

	def update_window(self):
		x = (self.window.get_width() - self.camera.resolution[0]) / 2
		y = (self.window.get_height() - self.camera.resolution[1]) / 2

		rgb = bytearray(self.camera.resolution[0] * self.camera.resolution[1] * 3)

		stream = io.BytesIO()
		self.camera.capture(stream, use_video_port=True, format='rgb')
		stream.seek(0)
		stream.readinto(rgb)
		stream.close()

		img = pygame.image.frombuffer(rgb[0:(self.camera.resolution[0] * self.camera.resolution[1] * 3)], self.camera.resolution, 'RGB')
		self.window.fill(0)

		if img:
			self.window.blit(img, (x,y))

		pygame.display.update()

	def write_csv(self, data, directory):
		with open(directory + "labels.csv", 'a', encoding='UTF8') as f:
			writer = csv.writer(f)
			writer.writerow(data)

	def get_image(self):
		self.imageFrame = pygame.surface.Surface((640, 480),0,self.window)
		self.process_surface_image()

                # resize and convert to grey scale
		self.transform_grey_scale()

		self.image_resize()

	def data_capture(self, input, number, distance, directory):
		self.camera.capture(directory + 'image{}.jpg'.format(number), resize=(200, 100))

		csvData = ["image{}.jpg".format(number), distance, input]

		self.write_csv(csvData, directory)

	# resize and transform on capture to allow using high resolution for viewing cam stream in window still
	def image_resize(self):
		self.imageFrame = cv2.resize(self.imageFrame, self.resizeDims)

	def transform_grey_scale(self):
		self.imageFrame = cv2.cvtColor(self.imageFrame, cv2.COLOR_RGB2GRAY)

	def release(self):
		self.loopFlag = False
		self.camera.close()
		pygame.quit()
