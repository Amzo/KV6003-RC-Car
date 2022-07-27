#!/usr/bin/env python
import time

try:
    from gpiozero import Motor, AngularServo
except RuntimeError:
    print("Error importing GPIO module")


class Car:
    def __init__(self):
        # keep pwm disabled for now as we have no need for it
        self.__rearLeftWheel = Motor(26, 19)
        self.__frontRightWheel = Motor(23, 24)
        self.__frontLeftWheel = Motor(6, 13)
        self.__rearRightWheel = Motor(7, 16)

    def frw(self, direction):
        if direction:
            self.__frontRightWheel.forward()
        else:
            self.__frontRightWheel.backward()

    def flw(self, direction):
        if direction:
            self.__frontLeftWheel.forward()
        else:
            self.__frontLeftWheel.backward()

    def rrw(self, direction):
        if direction:
            self.__rearRightWheel.forward()
        else:
            self.__rearRightWheel.backward()

    def rlw(self, direction):
        if direction:
            self.__rearLeftWheel.forward()
        else:
            self.__rearLeftWheel.backward()

    def move_forward(self):
            self.frw(True)
            self.flw(True)
            self.rrw(True)
            self.rlw(True)

    def move_backwards(self):
            self.frw(False)
            self.flw(False)
            self.rrw(False)
            self.rlw(False)

    def turn_left(self):
            self.frw(True)
            self.flw(False)
            self.rrw(True)
            self.rlw(False)
            # Move wheels in opposing directions to turn

    def turn_right(self):
            self.frw(False)
            self.flw(True)
            self.rrw(False)
            self.rlw(True)

    # the time to 90 degrees depends entirely on the battery charge
    # turns quicker at full power and slower at lower levels
    def turn_right_90(self):
        self.turn_right()
        time.sleep(0.72)

    def turn_left_90(self):
        self.turn_left()
        time.sleep(0.72)

    def release(self):
        self.__frontRightWheel.stop()
        self.__frontLeftWheel.stop()
        self.__rearRightWheel.stop()
        self.__rearLeftWheel.stop()


class Servo(AngularServo):
    def __init__(self, pin, defaultAngle):
        super().__init__(pin, min_angle=-90, max_angle=90)
        self.angle = defaultAngle

    def turn_motor(self, increment):
        if self.ceiling_check(increment):
            self.angle += increment

    def ceiling_check(self, increment):
        if -90 < (self.angle + increment) <= 90:
            return True
        else:
            return False
