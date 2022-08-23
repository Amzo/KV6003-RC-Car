import io
import shutil
import socket
import socketserver
import struct
from tkinter import messagebox

import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from lib.debug import LogInfo


class VideoStreamHandler(socketserver.StreamRequestHandler):
    def handle(self, connection, commands, tabGui, rootGui):
        try:
            connection = self.clientSocket.makefile('rb')
        except:
            print("Connection Failed")

        finally:
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

                # As the image comes in streams, ensure we don't update the image while a new image
                # is being fetched
                if not rootGui.newFrame:
                    rootGui.checkFrame = ImageTk.PhotoImage(image=Image.fromarray(imageRGB))
                    rootGui.predFrame = Image.fromarray(imageRGB)
                    tabGui.imageCount.value += 1
                    rootGui.newFrame = True

                if tabGui.gotPrediction.value == 1:
                    objResult = tabGui.objResults.value.decode()
                    print("Sending command {}".format(tabGui.results.value.decode()))
                    commands.send(('{}{}\n'.format(objResult, tabGui.results.value.decode()).encode('utf-8')))
                    tabGui.gotPrediction.value = 0


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
        try:
            self.clientSocket.connect((self.host, self.port))
        except OSError:
            messagebox.showwarning('Warning', 'Failed to connect to host')
        else:
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
