#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:44:50 2022

@author: Anthony Donnelly
"""
import socket, picamera, time, struct, sys, io
import threading
class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)
        self.stream.write(buf)
		
class Server(object):
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.videoThread = None
		self.activeConnection = True

	def start(self):
		while True:
			if self.videoThread is None:
				self.videoThread = threading.Thread(target=self.videoStream, args=(), daemon=True)
				self.videoThread.start()
			elif not self.videoThread.is_alive():
				self.videoThread = None

	def videoStream(self):
		self.serverSocket = socket.socket()
		self.serverSocket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
		print("binding socket to ", self.host, self.port)
		self.serverSocket.bind((self.host, self.port))

		print("Listening for connections")
		self.serverSocket.listen(1)

		try:
			print("Waiting for Connection")
			self.connection,self.client_address = self.serverSocket.accept()
			self.connection=self.connection.makefile('wb')
		except:
			pass

		print("Closing server socket")
		self.serverSocket.close()

		try:
			output = SplitFrames(self.connection)
# 			with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
# 				camera.resolution = (640,480)      # pi camera resolution
# 				camera.framerate = 60               # 15 frames/sec
# 				time.sleep(2)                       # give 2 secs for camera to initilize
# 				start = time.time()
# 				stream = io.BytesIO()
#                 # send jpeg format video stream
# 				print ("Start transmit ... ")

# 				for image in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
# 					try:
# 						self.connection.flush()
# 						stream.seek(0)
# 						b = stream.read()
# 						length=len(b)
# 						if length >5120000:
# 							continue
# 						lengthBin = struct.pack('L', length)
# 						self.connection.write(lengthBin)
# 						self.connection.write(b)
# 						stream.seek(0)
# 						stream.truncate()
# 					except Exception as e:
# 						print(e)
# 						print ("End transmit ... " )
# 						break
			with picamera.PiCamera(resolution='VGA', framerate=24) as camera:
				time.sleep(2)
				while self.activeConnection:
					camera.start_recording(output, format='mjpeg')
					camera.wait_recording(30 * 60)
					camera.stop_recording()
					


			self.connection.write(struct.pack('<L', 0))
		except:
			pass
