#!/usr/bin/env python

# simple check to avoid going out of bounds on the servo
def ceilingCheck(increment):
	if increment > -90 and increment <= 90:
		return True
	else:
		return False

def turnMotor(motor, increment):
	if ceilingCheck(increment):
		motor.angle = (increment)
