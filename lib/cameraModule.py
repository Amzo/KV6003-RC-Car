##!/usr/bin/env python3
import threading

import csv
import cv2
import io
import picamera
import pygame
from time import sleep

def write_csv(data, directory):
    with open(directory + "labels.csv", 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)


class carCamera:
    def __init__(self):
        pygame.init()
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)
        self.outputLocation = None
        
        self.capture = False
        self.imageNumber = None
        self.imageFrame = None
        self.window = pygame.display.set_mode((640, 480))
        self.loopFlag = True

        self.cameraThread = None

    def start(self):
        self.cameraThread = threading.Thread(target=self.update_window, args=(), daemon=True)
        self.cameraThread.start()

    def update_window(self):
        x = (self.window.get_width() - self.camera.resolution[0]) / 2
        y = (self.window.get_height() - self.camera.resolution[1]) / 2

        rgb = bytearray(self.camera.resolution[0] * self.camera.resolution[1] * 3)

        stream = io.BytesIO()
        self.camera.capture(stream, use_video_port=True, format='rgb')
        stream.seek(0)
        stream.readinto(rgb)
        stream.close()

        img = pygame.image.frombuffer(rgb[0:(self.camera.resolution[0] * self.camera.resolution[1] * 3)],
                                      self.camera.resolution, 'RGB')
        self.window.fill(0)

        if img:
            self.window.blit(img, (x, y))

        pygame.display.update()

    def data_capture(self, keyInput, number, distance, prevKey, directory):
        self.camera.capture(directory + "/" + keyInput + '/image{}.jpg'.format(number), resize=(320, 240))

        csvData = ["{}/image{}.jpg".format(keyInput, number), distance, prevKey, keyInput]

        write_csv(csvData, directory)

    def transform_grey_scale(self):
        self.imageFrame = cv2.cvtColor(self.imageFrame, cv2.COLOR_RGB2GRAY)

    def release(self):
        self.loopFlag = False
        self.camera.close()
        pygame.quit()
