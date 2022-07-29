#!/usr/bin/env python3
import glob
import hashlib
import os
import pickle
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, StringVar

import numpy as np
from PIL import ImageTk, ImageDraw, Image, UnidentifiedImageError, ImageFont

from lib.debug import LogInfo
from lib import predictTab, menuBar, debug, trainTab
from filelock import FileLock


class TabWidget:
    def __init__(self, master=None):
        self.tabControl = ttk.Notebook(master)
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='Predict')
        self.tabControl.add(self.tab2, text='Train')
        self.tabControl.pack(expand=1, fill="both")


class Gui(threading.Thread):
    dataCount: StringVar

    def __init__(self):
        threading.Thread.__init__(self)
        self.loadingImage = False
        self.frameCount = 0
        self.ts = None
        self.piWindow = None
        self.trainTab = None
        self.tab = None
        self.dataTab = None
        self.predictTab = None
        self.tabsObjects = None
        self.menuBar = None
        self.debugWindow = None
        self.debug = None
        self.model = None
        self.modelLoaded = None
        self.newFrame = False
        self.boxList = []
        if not os.path.exists('box.pkl'):
            self.hash = 1
        else:
            with open('box.pkl', 'rb') as f:
                self.hash = hashlib.md5(f.read()).hexdigest()
        self.passed = False

        # Frame data for each widget
        self.imgFrame = None
        self.predFrame = None
        self.checkFrame = None

        # calculate the remaining width of the window based on camera size to center widgets
        self.rootWindow = None
        self.remainingWidth = 880 - 640
        self.center = 880 - (self.remainingWidth / 2)
        self.centerTrain = 880 - ((880 - 570) / 2)
        self.start()

    def run(self):
        self.rootWindow = tk.Tk()
        self.rootWindow.title("KF6003: Individual Project")
        self.rootWindow.geometry("880x520")
        self.rootWindow.resizable(False, False)
        self.frameCount = 0

        self.debug = tk.BooleanVar(self.rootWindow)
        self.debug.set(False)

        self.modelLoaded = tk.BooleanVar(self.rootWindow)
        self.modelLoaded.set(False)

        self.menuBar = menuBar.MenuBar(master=self)

        self.tabsObjects = TabWidget(master=self.rootWindow)

        #########################################################################################
        #                                                                                       #
        #                                   Prediction tab widgets                              #
        #                                                                                       #
        #########################################################################################

        self.predictTab = predictTab.PredictTab(master=self, tabs=self.tabsObjects)
        self.predictTab.placeWidgets(self.center)

        #########################################################################################
        #                                                                                       #
        #                                    Training Widgets                                   #
        #                                                                                       #
        #########################################################################################

        self.trainTab = trainTab.TrainTab(master=self, tabs=self.tabsObjects)
        self.trainTab.placeWidgets()

        # event to handle tab change
        self.tabsObjects.tabControl.bind('<<NotebookTabChanged>>', self.tabChanged)
        self.rootWindow.protocol("WM_DELETE_WINDOW", self.onClosing)

        self.updateWindow()
        self.rootWindow.mainloop()

    def onClosing(self):
        files = glob.glob('data/*.jpg')
        for f in files:
            os.remove(f)
        self.rootWindow.quit()

    def enableDebug(self):
        self.debugWindow = debug.DebugWindow(master=self.rootWindow, main=self)

    def tabChanged(self, event):
        self.tab = event.widget.tab('current')['text']

    def updateWindow(self):
        if self.newFrame:
            self.imgFrame = self.checkFrame

            # keep two copies one for prediction, one for azure, to avoid threads causing conflicts, or waiting for each
            # other to finish processing / writing / removing
            if not os.path.exists('image.jpg'):
                self.predFrame.save('image.jpg')

            if not os.path.exists('image1.jpg'):
                self.predFrame.save('image1.jpg')

            elif os.path.exists('box.pkl'):
                if not os.path.exists('filelock'):
                    Path('filelock').touch()
                    if self.debug:
                        self.debugWindow.logText(LogInfo.debug.value, 'File is locked (GUI process)')
                    with open('box.pkl', 'rb') as file:
                        hash = hashlib.md5(file.read()).hexdigest()

                        # reload changed file
                    if self.hash != hash:
                        try:
                            with open('box.pkl', 'rb') as file:
                                boxList = pickle.load(file)
                        except AttributeError as e:
                            print(e)
                            self.passed = False
                        else:
                            self.boxList = boxList
                            self.passed = True
                            self.hash = hash
                    os.remove('filelock')
                    if self.debug:
                        self.debugWindow.logText(LogInfo.debug.value, 'filelock is removed (GUI Process)')

                if self.passed or len(self.boxList) > 0:
                    draw = ImageDraw.Draw(self.predFrame)
                    lineWidth = int(np.array(self.predFrame).shape[1] / 100)
                    fnt = ImageFont.truetype("./data/arial.ttf", 12)

                    for x in range(0, len(self.boxList)):
                        draw.line(self.boxList[x][0:5], fill=self.boxList[x][-3], width=lineWidth)
                        draw.text(self.boxList[x][0],
                                  self.boxList[x][-2] + ": {0:.2f}%".format(self.boxList[x][-1] * 100),
                                  font=fnt, fill=self.boxList[x][-3])

                    self.imgFrame = ImageTk.PhotoImage(image=self.predFrame)

            self.predictTab.videoPredLabel.configure(image=self.imgFrame)
            self.newFrame = False

        self.rootWindow.after(1, self.updateWindow)
