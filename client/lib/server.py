import io
import shutil
import socket
import socketserver
import struct
import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from lib import models as ourModel
from lib.debug import LogInfo
import lib.azure as carObject


class VideoStreamHandler(socketserver.StreamRequestHandler):
    def handle(self, connection, commands, tabGui, rootGui):
        try:
            connection = self.clientSocket.makefile('rb')
        except:
            print("Connection Failed")

        finally:
            imageCount = 0
            detectionCount = 0
            carObjectDetect = carObject.CarObjectDetection()
            detectionPerformed = False

            while self.connectFlag:
                imageLength = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

                if not imageLength:
                    break

                imageStream = io.BytesIO()
                imageStream.write(connection.read(imageLength))
                imageStream.seek(0)

                fileBytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)

                image = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(imageRGB)

                rootGui.predFrame = ImageTk.PhotoImage(image=Image.fromarray(imageRGB))

                rootGui.updateWindow()
                imageCount += 1

                if rootGui.predictTab.modelLoaded is True and not detectionPerformed:
                    carObjectDetect.checkImage = image
                    carObjectDetect.getPrediction()

                    carObjectDetect.filterResults()

                    for x in carObjectDetect.results_list:
                        print("Sending command {}".format(x))
                        commands.send(('{0}\n'.format(x)).encode('utf-8'))

                    detectionPerformed = True
                    detectionCount = 0

                if tabGui.selectedModel.get() == "CNN" and imageCount >= 6 \
                        and tabGui.modelLoaded:
                    rootGui.checkFrame = image

                    imageCount = 0
                    detectionCount += 1
                    tabGui.ourModel.makePrediction(check_frame=rootGui.checkFrame)

                    if rootGui.debug.get():
                        rootGui.debugWindow.logText(LogInfo.debug.value,
                                                    "Got prediction {}".format(tabGui.ourModel.results[0]))

                    print("Sending command {}".format(tabGui.ourModel.results[0]))
                    commands.send(('{0}\n'.format(tabGui.ourModel.results[0])).encode('utf-8'))

                if detectionCount >= 12:
                    detectionPerformed = False


class Server:
    def __init__(self, host, port, tabGui, rootGui):
        self.connectFlag = None
        self.tabs = tabGui
        self.rootGui = rootGui
        self.host = host
        self.port = port
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.commandSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def video_stream(self, videostream):
        self.clientSocket.connect((self.host, self.port))
        self.commandSocket.connect((self.host, 8080))

        self.connectFlag = True
        videostream.handle(self, self.clientSocket, self.commandSocket, self.tabs, self.rootGui)

    def is_valid(self, buf):
        byteValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                byteValid = False
            else:
                try:
                    Image.open(io.BytesIO(buf)).verify()
                except:
                    byteValid = False
        return byteValid

    def run(self, videostream):
        self.video_stream(videostream)
