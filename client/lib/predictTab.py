import os.path
import pickle
import threading
from datetime import time
from functools import partial

import tkinter as tk
from tkinter import messagebox, filedialog

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from lib import piWindow
from lib import server as ourServer
from lib.debug import LogInfo
from lib import models as OurModels
import lib.azure as carObject
import multiprocessing


class PredictTab:
    def __init__(self, master=None, tabs=None):
        self.objResults = multiprocessing.Value('c', b't')
        self.imageCount = multiprocessing.Value('i', 0)
        self.results = multiprocessing.Value('c', b't')
        self.gotPrediction = multiprocessing.Value('i', 0)
        self.results_list = None
        self.ourModel = OurModels.CustomModel(root_window=master)
        self.model = None
        self.modelLoaded = multiprocessing.Value('b', False)
        self.loaded_model = None
        self.up = None
        self.ts = None
        self.mainWindow = None
        self.videoStream = None
        self.rootClass = master
        self.up = tk.StringVar(master.rootWindow)
        self.down = tk.StringVar(master.rootWindow)
        self.left = tk.StringVar(master.rootWindow)
        self.right = tk.StringVar(master.rootWindow)
        self.forward = tk.StringVar(master.rootWindow)
        self.reverse = tk.StringVar(master.rootWindow)
        self.turnLeft = tk.StringVar(master.rootWindow)
        self.turnRight = tk.StringVar(master.rootWindow)
        self.connectText = tk.StringVar(master.rootWindow)
        self.pauseText = tk.StringVar(master.rootWindow)
        self.keyMap = {}
        self.move = multiprocessing.Value('i', 0)

        self.selectedModel = tk.StringVar(master.rootWindow)
        self.modelList = ['CNN']
        self.selectedModel.set(self.modelList[0])

        self.modelLocation = tk.StringVar(master.rootWindow)

        self.connectFrame = tk.LabelFrame(tabs.tab1, text="Car Connection", height=130, width=220)
        self.modelFrame = tk.LabelFrame(tabs.tab1, text="Model Select", height=70, width=220)

        self.videoPredLabel = tk.Label(tabs.tab1, borderwidth=3)
        self.hostLbl = tk.Label(tabs.tab1, text="Host:", font="Helvetica 9 bold")
        self.hostIp = tk.Entry(tabs.tab1, width=18)
        self.portLbl = tk.Label(tabs.tab1, text="Port:", font="Helvetica 9 bold")
        self.port = tk.Entry(tabs.tab1, width=18)

        self.modelSelect = tk.Label(tabs.tab1, text="Model:", font="Helvetica 9 bold")
        self.modelDropDown = tk.OptionMenu(tabs.tab1, self.selectedModel, *self.modelList)

        self.modelBrowseButton = tk.Button(tabs.tab1, text="Browse", command=self.modelBrowse)
        self.moveButton = tk.Button(tabs.tab1, text="Move", command=self.moveCar)
        self.pauseText.set("Pause")
        self.stopButton = tk.Button(tabs.tab1, textvariable=self.pauseText, command=self.stop)
        self.connectText.set("Connect")
        self.connectButton = tk.Button(tabs.tab1, textvariable=self.connectText,
                                       command=partial(self.connect))

    def placeWidgets(self, placement):
        self.videoPredLabel.place(x=640 / 2, y=490 / 2, anchor=tk.CENTER)
        self.connectFrame.place(x=650, y=5)

        self.hostLbl.place(x=placement - 50, y=40, anchor=tk.E)
        self.hostIp.place(x=placement - 30, y=40, anchor=tk.W)
        self.portLbl.place(x=placement - 50, y=70, anchor=tk.E)
        self.port.place(x=placement - 30, y=70, anchor=tk.W)
        self.connectButton.place(x=placement, y=110, anchor=tk.CENTER)

        self.modelFrame.place(x=650, y=235)
        self.modelSelect.place(x=placement - 60, y=275, anchor=tk.E)
        self.modelDropDown.place(x=placement + 20, y=275, anchor=tk.E)
        self.modelBrowseButton.place(x=placement + 35, y=275, anchor=tk.W)
        self.moveButton.place(x=placement + 50, y=340, anchor=tk.CENTER)
        self.stopButton.place(x=placement - 50, y=340, anchor=tk.CENTER)

    def connect(self):
        self.mainWindow = self.rootClass
        try:
            h = str(self.hostIp.get())
        except ValueError:
            messagebox.showerror('Error', 'Please enter a valid IP Address')
        else:
            try:
                p1 = int(self.port.get())
            except ValueError:
                messagebox.showerror('Error', 'Please enter a valid port')
            else:
                self.videoStream = ourServer.VideoStreamHandler
                self.ts = ourServer.Server(h, p1, self, self.mainWindow)
                T = threading.Thread(target=self.ts.run, args=(self.videoStream,), daemon=True)
                T.start()

    # don't block other threads
    def predictionThread(self):
        carObjectDetect = carObject.CarObjectDetection()
        self.loaded_model = self.ourModel.loadModel()
        self.modelLoaded.value = True

        while True:
            if os.path.exists('image1.jpg'):
                try:
                    im = Image.open('image1.jpg')
                except UnidentifiedImageError:
                    # image hasn't finished saving?
                    pass
                finally:
                    carObjectDetect.getPrediction()
                    self.objResults.value = carObjectDetect.filterResults(im)

            if self.move.value == 1:
                self.gotPrediction.value = 0
            else:
                if self.imageCount.value >= 12:
                    if os.path.exists('image.jpg'):
                        im = Image.open(r"image.jpg")

                    if self.gotPrediction.value == 0:
                        result = self.ourModel.makePrediction(im)
                        self.results.value = str.encode(result)

                        if self.rootClass.debug.get():
                            self.rootClass.debugWindow.logText(LogInfo.debug.value,
                                                               "Got prediction {}".format(self.results.value))

                        self.gotPrediction.value = 1
                        self.imageCount.value = 0
                        os.remove('image.jpg')

    def modelBrowse(self):
        filetypes = (
            ('Pickle files', '*.pkl'),
        )

        if self.selectedModel.get() == "CNN":
            self.modelLocation.set(filedialog.askdirectory())

            if os.path.isdir(self.modelLocation.get() + "/assets") and os.path.isdir(
                    self.modelLocation.get() + "/variables"):
                if self.rootClass.debug.get():
                    self.rootClass.debugWindow.logText(LogInfo.debug.value, "Found CNN Model")

    def stop(self):
        if self.move.value == 0:
            self.move.value = 1
            self.pauseText.set("Continue")
        else:
            self.move.value = 0
            self.pauseText.set("Pause")

    def moveCar(self):
        try:
            self.ts.connectFlag
        except AttributeError:
            messagebox.showerror('Error', 'Connect to the remote car server')
        else:
            print("starting process")
            ps = multiprocessing.Process(target=self.predictionThread)
            ps.start()
