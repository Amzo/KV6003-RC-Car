#!/usr/bin/env python3

def parseKeyboard(event, rcCar, servoDirection, servorIncrement):
	if event.type == pygame.KEYUP:
		rcCar.stop()
	if event.type == pygame.KEYDOWN:
		# get distance before executing a movement
		distanceSetup.getDistance(rcDistance)
		if event.key == pygame.K_w:
			rcCar.moveForward()
		elif event.key == pygame.K_s:
			rcCar.moveBackwards()
		elif event.key == pygame.K_a:
			rcCar.turnLeft()
		elif event.key == pygame.K_d:
			rcCar.turnRight()
		elif event.key == pygame.K_LEFT:
			leftRight += 10
			servo.turnMotor(servoDirection, leftRight)
		elif event.key == pygame.K_RIGHT:
			leftRight -= 10
			servo.turnMotor(servoLeftRight, leftRight)
		elif event.key == pygame.K_UP:
			upDown -= 10
			servo.turnMotor(servoUpDown, upDown)
		elif event.key == pygame.K_DOWN:
			upDown += 10
			servo.turnMotor(servoUpDown, upDown)
		elif event.key == pygame.K_q:
			# reset everything and exit
			rcCar.stop()
