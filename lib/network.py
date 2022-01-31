#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:44:50 2022

@author: Anthony Donnelly
"""
import socket, picamera, time, struct, sys, io
import threading

class Server(object):
	def __init__(self, host, port):
		self.host = host
		self.port = port

		self.serverSocket = socket.socket()
		self.serverSocket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
		print("binding socket to ", self.host, self.port)
		self.serverSocket.bind((self.host, self.port))
		print("Listening for connections")
		self.serverSocket.listen(1)


	def start(self):
		self.videoThread = threading.Thread(target=self.videoStream, args=(), daemon=True)
		self.videoThread.start()

	def videoStream(self):
		try:
			print("Waiting for Connection")
			self.connection,self.client_address = self.serverSocket.accept()
			self.connection=self.connection.makefile('wb')
		except:
			pass

		print("Closing server socket")
		self.serverSocket.close()

		try:
			with picamera.PiCamera() as camera:
				camera.resolution = (640,480)      # pi camera resolution
				camera.framerate = 15               # 15 frames/sec
				time.sleep(2)                       # give 2 secs for camera to initilize
				start = time.time()
				stream = io.BytesIO()
                # send jpeg format video stream
				print ("Start transmit ... ")

				for image in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
					try:
						self.connection.flush()
						stream.seek(0)
						b = stream.read()
						length=len(b)
						if length >5120000:
							continue
						lengthBin = struct.pack('L', length)
						self.connection.write(lengthBin)
						self.connection.write(b)
						stream.seek(0)
						stream.truncate()
					except Exception as e:
						print(e)
						print ("End transmit ... " )
						break
		except:
            #print "Camera unintall"
			pass
