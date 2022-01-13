#!/usr/bin/env python

try:
	import RPi.GPIO as GPIO
except RuntimeError:
	print("Error importing GPIO module")

class Car:
	def __init__(self):
		GPIO.setmode(GPIO.BOARD)
		GPIO.setwarnings(False)
		# if both pins are high or low, motor won't spin
		# if one pin is high, and one is low, motor will spin
		# in the corresponding direction
		self.rearLeftWheel = [37, 35]
		self.frontRightWheel = [16, 18]
		self.frontLeftWheel = [31, 33]
		self.rearRightWheel = [26, 36]

	def moveForward(self):
		GPIO.output(self.frontRightWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.frontLeftWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.rearRightWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.rearLeftWheel, (GPIO.HIGH, GPIO.LOW))

	def moveBackwards(self):
		GPIO.output(self.frontRightWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.frontLeftWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.rearRightWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.rearLeftWheel, (GPIO.LOW, GPIO.HIGH))

	def turnLeft(self):
		# Move wheels in opposing directions to turn
		GPIO.output(self.frontRightWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.frontLeftWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.rearRightWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.rearLeftWheel, (GPIO.LOW, GPIO.HIGH))

	def turnRight(self):
		GPIO.output(self.frontRightWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.frontLeftWheel, (GPIO.HIGH, GPIO.LOW))
		GPIO.output(self.rearRightWheel, (GPIO.LOW, GPIO.HIGH))
		GPIO.output(self.rearLeftWheel, (GPIO.HIGH, GPIO.HIGH))



	def pinReset(self):
		GPIO.setup(self.frontLeftWheel, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.frontRightWheel, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.rearLeftWheel, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.rearRightWheel, GPIO.OUT, initial=GPIO.LOW)


