# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:46:01 2022

@author: Anthony Donnelly
"""
import socketserver, cv2
import numpy as np
import struct, socket, pygame, time
from threading import Thread
import io
from PIL import Image

class VideoStreamHandler(socketserver.StreamRequestHandler):
	def handle(self, connection, window):
		streamBytes = b' '

		try:
			connection = self.clientSocket.makefile('rb')
		except:
			print("Connection Failed")
		finally:
			while self.connectFlag:
				try:
					streamBytes= connection.read(4)
					len=struct.unpack('<L', streamBytes[:4])
					jpg=connection.read(len[0])

					if self.is_valid(jpg):
						window.image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

				except Exception as e:
					print (e)
					break

class PyWindow():
	def __init__(self):
		pygame.init()
		self.window = pygame.display.set_mode((640, 480))
		self.x = (self.window.get_width() - 640) / 2
		self.y = (self.window.get_height() - 480) / 2
		self.image = None

	def update(self):
		pygame.event.pump

		if self.image is not None:
			img = pygame.image.frombuffer(self.image.tobytes(), self.image.shape[1::-1], "RGB")
			self.window.fill(0)

			if img:
				self.window.blit(img, (self.x,self.y))

			pygame.display.update()

	def run(self):
		while True:
			self.update()

class Server():
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def video_stream(self, videostream):
		self.clientSocket.connect((self.host, self.port))
		self.connectFlag=True
		videostream.handle(self, self.clientSocket, display)

	def is_valid(self, buf):
		byteValid = True
		if buf[6:10] in (b'JFIF', b'Exif'):
			if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
				byteValid = False
			else:
				try:
					Image.open(io.BytesIO(buf)).verify()
				except:
					byteValid = False
		return byteValid

	def run(self, videostream):
		self.video_stream(videostream)


if __name__ == '__main__':
	h, p1 = "192.168.0.63", 50022
	videoStream = VideoStreamHandler

	display = PyWindow()
	ts = Server(h, p1)

	T = Thread(target=ts.run, args=(videoStream,), daemon=True)
	T.start()
	T2 = Thread(target=display.run)
	T2.start()

	time.sleep(100)
