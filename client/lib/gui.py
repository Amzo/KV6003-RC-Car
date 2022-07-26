#!/usr/bin/env python3
import glob
import os
import threading
import tkinter as tk
from io import BytesIO
from os.path import exists
from tkinter import ttk, StringVar

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib import predictTab, menuBar, debug, trainTab


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

            if not os.path.exists('image.jpg'):
                print('saving')
                self.predFrame.save('image.jpg')

            self.predictTab.videoPredLabel.configure(image=self.imgFrame)
            self.newFrame = False

        self.rootWindow.after(1, self.updateWindow)


