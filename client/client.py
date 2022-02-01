# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:46:01 2022

@author: Anthony Donnelly
"""
import socketserver, cv2
import numpy as np
import struct, socket
import threading
import io
from PIL import Image

class VideoStreamHandler(socketserver.StreamRequestHandler):
	def handle(self, connection):
		windowName="Pi Camera Stream"

		try:
			connection = self.clientSocket.makefile('rb')
		except:
			print("Connection Failed")
		finally:
			while self.connectFlag:
				try:

					imageLength = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
					
					if not imageLength:
						break
					
					imageStream = io.BytesIO()
					imageStream.write(connection.read(imageLength))
					imageStream.seek(0)
					
					fileBytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)
					
					image = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)
					
					cv2.imshow(windowName, image)
					cv2.waitKey(1)
					
					if cv2.getWindowProperty(windowName,cv2.WND_PROP_VISIBLE) < 1:
							self.connectFlag = False
						
				except Exception as e:
					print (e)
					self.connectFlag = False
					
class ImageFeature():
	def derp(self):
		print('detectFace')
	
		print('detectStop')
	
class Server():
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def video_stream(self, videostream):
		self.clientSocket.connect((self.host, self.port))
		self.connectFlag=True
		videostream.handle(self, self.clientSocket)

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
	ts = Server(h, p1)

	T = threading.Thread(target=ts.run, args=(videoStream, ), daemon=True)
	T.start()


