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
import numpy
import pickle
import time

# A bug in spyder that prevents the kernel from restarting
# when importing tensorflow and using logging
# https://github.com/spyder-ide/spyder/issues/13644
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from mtcnn import MTCNN
from tensorflow.config import experimental
import tensorflow as tf

class VideoStreamHandler(socketserver.StreamRequestHandler):
	def handle(self, connection, commands):
		windowName="Pi Camera Stream"
		faceDetect = False
		#detector = MTCNN()
		frameCount = 0
		lastPred = None
		model, encoder = loadModel()
		
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

					if frameCount >= 8:
						pred = getPrediction(model, encoder, image)
						print(pred)
						if (lastPred != "e" and lastPred != "q" or pred != lastPred):
							commands.send((pred + '\n').encode('utf-8'))
							if pred == "e" or pred == "q":
								lastPred = pred
						frameCount = 0
					#image = detectArrow(image)
										
					# we get on average 24 fps. performing this many detections on each frame is costly
					# reduce strain by performing detection on every few frames, our eyes shouldn't be able to notice
					# even with a GPU accelleration of face detection is slow on my laptop
					# frameCount value may exceed 4 if we don't enable faceDetection with f key
					if faceDetect and frameCount >= 2:
						print("detecting face")
						image = detectFace(image)
						#image = detectFace(image)
						frameCount = 0
						
					cv2.imshow(windowName, image)
					k = cv2.waitKey(1)
					frameCount += 1
					
					#toggle face detection on f key
					if k == 102:
						faceDetect = not faceDetect
					
					if cv2.getWindowProperty(windowName,cv2.WND_PROP_VISIBLE) < 1:
							self.connectFlag = False
						
				except Exception as e:
					print (e)
					self.connectFlag = False
					
class Server():
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.commandSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def video_stream(self, videostream):
		self.clientSocket.connect((self.host, self.port))
		self.commandSocket.connect((self.host, 8080))
		self.connectFlag=True
		videostream.handle(self, self.clientSocket, self.commandSocket)

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

def detectFace(frame):
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	frame = cv2.resize(frame, (100,50), interpolation = cv2.INTER_AREA)
	greyImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(greyImage, 1.3, 5)
	
	for (x,y,w,h) in faces:
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	frame = cv2.resize(frame,(640,480), interpolation = cv2.INTER_AREA)
	
	return frame

def detectArrow(frame):
	arrowCascade = cv2.CascadeClassifier('xml/arrow-classifier.xml')
	#frame = cv2.resize(frame, (200,100), interpolation = cv2.INTER_AREA)
	greyImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	arrowDetected = arrowCascade.detectMultiScale(
        greyImage, minNeighbors=6, minSize=(30, 30))
	for (x, y, w, h) in arrowDetected:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
		
	return frame

def detectFaceCNN(detector, frame):

	boxes = detector.detect_faces(frame)
	
	if boxes:
		box = boxes[0]['box']
		conf = boxes[0]['confidence']
		x, y, w, h = box[0], box[1], box[2], box[3]
 
		if conf > 0.5:
			frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
			
		return frame
	
def loadModel():
	pkl_file = open('models/classes.pkl', 'rb')
	encoder = pickle.load(pkl_file) 

	model = tf.keras.models.load_model('models/carModel.h5')
	
	return model, encoder


def getPrediction(model, encoder, frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(gray,(200,100), interpolation = cv2.INTER_AREA)
	img = frame/255
	#img = np.expand_dims(img, axis=0)
	frame = img[:, :, np.newaxis]
	frame = np.expand_dims(frame, axis=0)
	prediction = model.predict(frame)
	predicted =  encoder.inverse_transform(prediction)
	
	return predicted[0]

if __name__ == '__main__':
	physical_devices = experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	config = experimental.set_memory_growth(physical_devices[0], True)
	
	h, p1 = "192.168.0.63", 50022
	videoStream = VideoStreamHandler
	ts = Server(h, p1)

	T = threading.Thread(target=ts.run, args=(videoStream,), daemon=True)
	T.start()


