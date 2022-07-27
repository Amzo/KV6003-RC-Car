##!/usr/bin/env python3
import os

import picamera
import pygame


class CarCamera:
    def __init__(self):
        self.saveDirectory = str
        self.key = str
        self.imageNumber = None
        self.writer = None
        pygame.init()
        width: int = 320
        height: int = 240
        self.camera = picamera.PiCamera()
        # reduce capture size for rapid realtime frame capture
        self.camera.resolution = (width, height)
        self.window = pygame.display.set_mode((width, height))

        # data capture
        self.saveDirectory: str
        self.key: str
        self.imageNumber: int

        self.csvFile = None
        self.prevKey: float = 0.0

    def release(self):
        pygame.quit()

    def data_capture(self):
        # f strings are by far the fastest way for string manipulation in pure python
        # but they cause optimization issues in cython, use join instead, ugly but faster
        # and speed is needed here to capture as many images as we can within each key press
        cdef list joinList = [self.saveDirectory, self.key]
        cdef str outputDir = ''.join(joinList)
        cdef str number = str(self.imageNumber)
        if not os.path.exists(str(outputDir)) and self.key is not None:
                os.makedirs(str(outputDir))
        elif self.key is not None and os.path.exists(str(outputDir)):
                joinList.append("/")
                joinList.append("image")
                joinList.append(number)
                joinList.append(".jpg")
                print(''.join(joinList))
                self.camera.capture(''.join(joinList), use_video_port=True)
                csvData = [f"{self.key}/image{number}.jpg", self.prevKey, self.key]
                self.writer.writerow(csvData)

                #self.imageNumber += 1

