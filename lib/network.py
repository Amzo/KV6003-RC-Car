#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:44:50 2022

@author: Anthony Donnelly
"""

import configparser
import io
import picamera
import socket
import struct
import threading
import time

config = configparser.ConfigParser()
config.read('config/config.ini')


def start_connection():
    # stream video remotely to reduce pi load
    streamConnection = Server(config['host']['serverName'], int(config['port']['Port']))
    print("Starting video streaming server")
    streamConnection.start()
    print("Running A.I")

    return streamConnection


class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.time = time.time()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))

                self.stream.seek(0)
        self.stream.write(buf)


class Server(object):
    def __init__(self, host, port):
        self.client_address = None
        self.clientAddressCommand = None
        self.connection = None
        self.serverSocket = None
        self.commandSocket = None
        self.commandThread = None
        self.host = host
        self.port = port
        self.videoThread = None
        self.activeConnection = True
        self.frame = None
        self.commands = None
        self.tcpFlag = False
        self.connectionCommand = None

    def start(self):
        self.videoThread = threading.Thread(target=self.videoStream, args=(), daemon=True)
        self.videoThread.start()

        self.commandThread = threading.Thread(target=self.commandStream, args=(), daemon=True)
        self.commandThread.start()

    def commandStream(self):
        self.commandSocket = socket.socket()
        self.commandSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.commandSocket.bind((self.host, 8080))

        self.commandSocket.listen(1)
        try:
            print("Waiting for connection to receive commands")
            self.connectionCommand, self.clientAddressCommand = self.commandSocket.accept()
        finally:
            pass
        print("Got connection for commands")
        self.commandSocket.close()
        self.tcpFlag = True

    def getCommand(self):
        while self.connectionCommand is None:
            # wait for a connection
            time.sleep(1)

        try:
            print("waiting for command")
            incomming = "" + self.connectionCommand.recv(3).decode('utf-8')
        # except:
        #    if self.tcpFlag:
        #        self.Reset()
        finally:
            pass

        if len(incomming) < 3:
            restCmd = incomming
            if restCmd == '' and self.tcpFlag:
                self.Reset()
        if incomming != '':
            self.commands = incomming.split("\n")
            if self.commands[-1] != "":
                self.commands = self.commands[:-1]

    def Reset(self):
        self.videoThread.join()
        self.commandThread.join()
        start_connection()

    def videoStream(self):
        self.serverSocket = socket.socket()
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        print("binding socket to ", self.host, self.port)
        self.serverSocket.bind((self.host, self.port))

        print("Listening for connections")
        self.serverSocket.listen(1)

        try:
            print("Waiting for Connection")
            self.connection, self.client_address = self.serverSocket.accept()
            self.connection = self.connection.makefile('wb')
        # except:
        #    pass
        finally:
            pass

        print("Closing server socket")
        self.serverSocket.close()

        try:
            output = SplitFrames(self.connection)

            with picamera.PiCamera(resolution=(640, 480), framerate=12) as camera:
                time.sleep(2)
                while self.activeConnection:
                    camera.start_recording(output, format='mjpeg')
                    camera.wait_recording(30 * 60)
                    camera.stop_recording()

            self.connection.write(struct.pack('<L', 0))
        # except:
        #    pass
        finally:
            pass
