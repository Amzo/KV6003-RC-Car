#!/usr/bin/env python

import pigpio

# simple check to avoid going out of bounds on the servo
def ceilingCheck(increment):
	if increment > 500 and increment <= 2500:
		return True
	else:
		return False

def turnMotor(motor, pin, increment):
	if ceilingCheck(increment):
		motor.set_servo_pulsewidth(pin, increment)
