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

	def moveForward(self):
		self.frontRightWheel.forward()
		self.frontLeftWheel.forward()
		self.rearRightWheel.forward()
		self.rearLeftWheel.forward()

	def moveBackwards(self):
		self.frontRightWheel.backward()
		self.frontLeftWheel.backward()
		self.rearRightWheel.backward()
		self.rearLeftWheel.backward()

	def turnLeft(self):
		# Move wheels in opposing directions to turn
		self.frontRightWheel.forward()
		self.frontLeftWheel.backward()
		self.rearRightWheel.forward()
		self.rearLeftWheel.backward()

	def turnRight(self):
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
        self.angle = 0;
        self = AngularServo(pin, min_angle=-90, max_angle=90)
        
    def turnMotor(self, increment):
        if self.ceilingCheck(increment):
            self.angle = self.angle + increment
            
    def ceilingCheck(self, increment):
        if self.angle + increment > -90 and self.angle  + increment<= 90:
            True
        else:
            False
