#!/usr/bin/env python
import ray
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


    #parallize the execution of the wheel. having the code linearly means some wheels start up before others
    # causing drifts to left or right, having each one run in parallel allongs the execution of all wheels
    # simutaneously
    @ray.remote
    def frw(self, direction):
        if direction:
            self.__frontRightWheel.forward()
        else:
            self.__frontRightWheel.backward()

    @ray.remote
    def flw(self, direction):
        if direction:
            self.__frontLeftWheel.forward()
        else:
            self.__frontLeftWheel.backward()

    @ray.remote
    def rrw(self, direction):
        if direction:
            self.__rearRightWheel.forward()
        else:
            self.__rearRightWheel.backward()

    @ray.remote
    def rlw(self, direction):
       if direction:
           self.__rearLeftWheel.forward()
       else:
           self.__rearLeftWheel.backward()

    def move_forward(self):
        ray.get([frw.remote(True), flw.remote(True), rrw.remote(True), rlw.remote(True)])

    def move_backwards(self):
        ray.get([frw.remote(False), flw.remote(False), rrw.remote(False), rlw.remote(False)])

    def turn_left(self):
        ray.get([frw.remote(True), flw.remote(False), rrw.remote(True), rlw.remote(False)])
        # Move wheels in opposing directions to turn

    def turn_right(self):
        ray.get([frw.remote(False), flw.remote(True), rrw.remote(False), rlw.remote(True)])

    # the time to 90 degrees depends entirely on the battery charge
    # turns quicker at full power and slower at lower levels
    def turn_right_90(self):
        self.turn_right()
        time.sleep(0.72)

    def turn_left_90(self):
        self.turn_left()
        time.sleep(0.72)

    def release(self):
        self.frontRightWheel.stop()
        self.frontLeftWheel.stop()
        self.rearRightWheel.stop()
        self.rearLeftWheel.stop()


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
