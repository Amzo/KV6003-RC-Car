#!/usr/bin/env python

import time

try:
	from gpiozero import Motor, AngularServo
except RuntimeError:
	print("Error importing GPIO module")

class Car():
	def __init__(self):
		# keep pwm disabled for now as we have no need for it
		self.rearLeftWheel = Motor(26, 19)
		self.frontRightWheel = Motor(23, 24)
		self.frontLeftWheel = Motor(6, 13)
		self.rearRightWheel = Motor(7, 16)

	def move_forward(self):
		self.frontRightWheel.forward(1)
		self.frontLeftWheel.forward(1)
		self.rearRightWheel.forward(1)
		self.rearLeftWheel.forward(1)

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
		
	# the time to 90 degrees depends entirely on the battery charge
	# turns quicker at full power and slower at lower levels		
	def turn_right_90(self):
		self.turn_right()
		time.sleep(0.72)
		
	def turn_left_90(self):
		self.turn_left()
		time.sleep(0.72)
		
	def release(self):
#		try:
		self.frontRightWheel.stop()
#		except:
#			# WE close the devices our selves, prevent gpio zero error from trying to close the already closed devices
#			pass
#		else:
		self.frontLeftWheel.stop()
		self.rearRightWheel.stop()
		self.rearLeftWheel.stop()

class Servo(AngularServo):
	def __init__(self, pin, defaultAngle):
		super().__init__(pin, min_angle=-90, max_angle=90)
		self.angle = defaultAngle;

	def turn_motor(self, increment):
		if self.ceiling_check(increment):
			self.angle += increment

	def ceiling_check(self, increment):
		if (self.angle + increment) > -90 and (self.angle  + increment) <= 90:
			return True
		else:
			return False
