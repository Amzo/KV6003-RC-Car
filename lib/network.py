#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:44:50 2022

@author: Anthony Donnelly
"""
import socket, picamera, time, struct, sys, io
import threading

# faster video streaming from piCamera documentation
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

		self.client_socket = socket.socket()
		self.client_socket.connect((self.host, self.port))
		self.connection = self.client_socket.makefile('wb')

	def start(self, res):
		self.videoThread = threading.Thread(target=self.videoStream, args=(res), daemon=True)
		self.videoThread.start()
		
	def videoStream(self, res):
		try:
			streamOutput = SplitFrames(self.connection)

			with picamera.PiCamera(resolution=res, framerate=30) as camera:
				time.sleep(2)
				start = time.time()
				camera.start_recording(streamOutput, format='mjpeg')
				camera.wait_recording(sys.maxint)
				camera.stop_recording()

				# Write the terminating 0-length to the connection to let the
				#server know we're done
				self.connection.write(struct.pack('<L', 0))
		finally:
			finish = time.time()
			print('Sent %d images in %d seconds at %.2ffps' % (
			streamOutput.count, finish-start, streamOutput.count / (finish-start)))
			self.connection.close()
			self.client_socket.close()
