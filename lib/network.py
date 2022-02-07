#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:44:50 2022

@author: Anthony Donnelly
"""
import socket, picamera, time, struct, io
import threading, numpy, cv2
from multiprocessing import Process

class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.time = time.time()
		
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

				# save 2 frames per second for model calculations
				# instead of real time video capture, saving on processing power
                # if (time.time() + 0.5) >= self.time:
                #        self.stream.seek(0)
                #        frame = numpy.fromstring(self.stream.read(size), dtype=numpy.uint8)
                #        frame = cv2.imdecode(frame, 1)
                #        frame = frame[:, :, ::-1]
                #        cv2.imwrite('testFrameSave.jpg', frame)
                #        self.time = time.time()
					   
                self.stream.seek(0)
        self.stream.write(buf)

class Server(object):
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.videoThread = None
		self.activeConnection = True
		self.frame = None
		self.commands = None
		self.tcpFlag = False
		self.connectionCommand = None
		
	def start(self):
		#self.videoProcess = Process(target=self.videoStream, args=())
		self.videoThread = threading.Thread(target=self.videoStream, args=(), daemon=True)
		self.videoThread.start()
		#self.videoProcess.start()

		self.commandThread = threading.Thread(target=self.commandStream, args=(), daemon=True)
		self.commandThread.start()
			
	def commandStream(self):
		self.commandSocket =  socket.socket()
		self.commandSocket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT, 1)
		self.commandSocket.bind((self.host, 8080))

		self.commandSocket.listen(1)
		try:
			print("Waiting for connection to receive commands")
			self.connectionCommand, self.clientAddressCommand = self.commandSocket.accept()
		except:
			pass
		
		print("Got connection for commands")
		self.commandSocket.close()
		self.tcpFlag = True
		
		#while True:
		#	self.getCommand()
			
	def getCommand(self):
		while self.connectionCommand is None:
			#wait for a connection
			time.sleep(1)
			
		try:
			print("waiting for command")
			incomming=""+self.connectionCommand.recv(3).decode('utf-8')
		except:
			if self.tcpFlag:
				self.Reset()

		if len(incomming) < 3:
			restCmd=incomming
			if restCmd=='' and self.tcpFlag:
				self.Reset()
			restCmd=""
		if incomming != '':
			self.commands=incomming.split("\n")
			if(self.commands[-1] != ""):
				restCmd=self.commands[-1]
				self.commands=self.commands[:-1]
						
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

			with picamera.PiCamera(resolution='VGA', framerate=24) as camera:
				time.sleep(2)
				while self.activeConnection:
					camera.start_recording(output, format='mjpeg')
					camera.wait_recording(30 * 60)
					camera.stop_recording()
					


			self.connection.write(struct.pack('<L', 0))
		except:
			pass
