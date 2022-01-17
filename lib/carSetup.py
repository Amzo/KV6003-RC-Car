#!/usr/bin/env python

try:
	from gpiozero import Motor, AngularServo
except RuntimeError:
	print("Error importing GPIO module")

class Car(Motor):
	def __init__(self):
		self.rearLeftWheel = Motor(26, 19)
		self.frontRightWheel = Motor(23, 24)
		self.frontLeftWheel = Motor(6, 13)
		self.rearRightWheel = Motor(7, 16)

	def move_forward(self):
		self.frontRightWheel.forward()
		self.frontLeftWheel.forward()
		self.rearRightWheel.forward()
		self.rearLeftWheel.forward()

	def move_backwards(self):
		self.frontRightWheel.backward()
		self.frontLeftWheel.backward()
		self.rearRightWheel.backward()
		self.rearLeftWheel.backward()

	def turn_left(self):
		# Move wheels in opposing directions to turn
		self.frontRightWheel.forward()
		self.frontLeftWheel.backward()
		self.rearRightWheel.forward()
		self.rearLeftWheel.backward()

	def turn_right(self):
		self.frontRightWheel.backward()
		self.frontLeftWheel.forward()
		self.rearRightWheel.backward()
		self.rearLeftWheel.forward()

	def stop(self):
		self.frontRightWheel.stop()
		self.frontLeftWheel.stop()
		self.rearRightWheel.stop()
		self.rearLeftWheel.stop()

class Servo(AngularServo):
	def __init__(self, pin):
		super().__init__(pin, min_angle=-90, max_angle=90)
		self.angle = 0;

	def turn_motor(self, increment):
		if self.ceiling_check(increment):
			self.angle += increment

	def ceiling_check(self, increment):
		if (self.angle + increment) > -90 and (self.angle  + increment) <= 90:
			return True
		else:
			return False
