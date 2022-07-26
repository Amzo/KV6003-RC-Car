import os.path
import pickle
import threading
from datetime import time
from functools import partial

import tkinter as tk
from tkinter import messagebox, filedialog

from PIL import Image

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
        self.keyMap = {}
        self.stop = False

        self.selectedModel = tk.StringVar(master.rootWindow)
        self.modelList = ['CNN']
        self.selectedModel.set(self.modelList[0])

        self.modelLocation = tk.StringVar(master.rootWindow)

        self.connectFrame = tk.LabelFrame(tabs.tab1, text="Car Connection", height=130, width=220)
        self.objectFrame = tk.LabelFrame(tabs.tab1, text="Object Detection", height=120, width=220)
        self.carFrame = tk.LabelFrame(tabs.tab1, text="Manual Car Control", height=110, width=220)
        self.modelFrame = tk.LabelFrame(tabs.tab1, text="Model Select", height=70, width=220)

        self.videoPredLabel = tk.Label(tabs.tab1, borderwidth=3)
        self.hostLbl = tk.Label(tabs.tab1, text="Host:", font="Helvetica 9 bold")
        self.hostIp = tk.Entry(tabs.tab1, width=18)
        self.portLbl = tk.Label(tabs.tab1, text="Port:", font="Helvetica 9 bold")
        self.port = tk.Entry(tabs.tab1, width=18)

        self.modelSelect = tk.Label(tabs.tab1, text="Model:", font="Helvetica 9 bold")
        self.modelDropDown = tk.OptionMenu(tabs.tab1, self.selectedModel, *self.modelList)

        self.modelBrowseButton = tk.Button(tabs.tab1, text="Browse", command=self.modelBrowse)
        self.moveButton = tk.Button(tabs.tab1, text="Move", command=self.move)
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

        self.objectFrame.place(x=650, y=135)

        self.carFrame.place(x=650, y=255)

        self.modelFrame.place(x=650, y=365)
        self.modelSelect.place(x=placement - 60, y=405, anchor=tk.E)
        self.modelDropDown.place(x=placement + 20, y=405, anchor=tk.E)
        self.modelBrowseButton.place(x=placement + 40, y=405, anchor=tk.W)
        self.moveButton.place(x=placement, y=460, anchor=tk.CENTER)

    def connect(self):
        self.mainWindow = self.rootClass
        h = str(self.hostIp.get())
        p1 = int(self.port.get())
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
            if self.imageCount.value >= 12 and self.modelLoaded.value:
                if os.path.exists('image.jpg'):
                    im = Image.open(r"image.jpg")

                carObjectDetect.getPrediction()
                carObjectDetect.filterResults()
                for x in carObjectDetect.results_list:
                    self.objResults = str.encode(x)

                if self.gotPrediction.value == 0:
                    if not self.stop:
                        result = self.ourModel.makePrediction(im)
                        self.results.value = str.encode(result)

                        if self.rootClass.debug.get():
                            self.rootClass.debugWindow.logText(LogInfo.debug.value,
                                                               "Got prediction {}".format(self.ourModel.results[0]))

                        print(self.results.value)

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

    def move(self):
        try:
            self.ts.connectFlag
        except AttributeError:
            messagebox.showerror('Error', 'Connect to the remote car server')

        ps = multiprocessing.Process(target=self.predictionThread)
        ps.start()

