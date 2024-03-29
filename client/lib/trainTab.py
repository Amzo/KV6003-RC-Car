import os
import pathlib
import tkinter as tk
from functools import partial
from threading import Thread
from tkinter import scrolledtext, ttk, filedialog

from lib.debug import LogInfo
from lib import models as ourModels


class TrainTab:
    def __init__(self, master=None, tabs=None):
        self.csvFound = False
        self.model = None
        self.csvFile = None
        self.saveLocation = None
        self.epochs = None
        self.rootWindow = master

        self.precision = tk.StringVar(master.rootWindow)
        self.recall = tk.StringVar(master.rootWindow)
        self.f1 = tk.StringVar(master.rootWindow)
        self.progressValue = tk.DoubleVar(master.rootWindow)
        self.saveLocation = tk.StringVar(master.rootWindow)
        self.csvFile = tk.StringVar(master.rootWindow)
        self.modelSaveLocation = tk.StringVar(master.rootWindow)

        # placement for precision under text box
        self.placement = (570 / 2.3)

        self.cnnFrame = tk.LabelFrame(tabs.tab2, text="CNN", height=355, width=250)

        self.outputWindowLabel = tk.Label(tabs.tab2, text="Learning Output:", font="Helvetica 10 bold")
        self.outputText = scrolledtext.ScrolledText(tabs.tab2, height=20, width=76, state=tk.DISABLED)
        self.accuracyLabel = tk.Label(tabs.tab2, text="Learning Accuracy:", font="Helvetica 10 bold")
        self.precisionLabel = tk.Label(tabs.tab2, text="Precision:", font="Helvetica 10 bold")
        self.precisionTxt = tk.Label(tabs.tab2, textvariable=self.precision, font="Helvetica 10 bold")
        self.recallLabel = tk.Label(tabs.tab2, text="recall:", font="Helvetica 10 bold")
        self.recallTxt = tk.Label(tabs.tab2, textvariable=self.recall, font="Helvetica 10 bold")
        self.F1Label = tk.Label(tabs.tab2, text="F1 Score:", font="Helvetica 10 bold")
        self.F1Txt = tk.Label(tabs.tab2, textvariable=self.f1, font="Helvetica 10 bold")

        # data input for training
        self.imageInputLabel = tk.Label(tabs.tab2, text="Image Folder", font="Helvetica 10 bold")
        self.imageInputText = tk.Entry(tabs.tab2, width=30)
        self.imageFolderButton = ttk.Button(tabs.tab2, text="Browse",
                                            command=partial(self.browseFolder, "directory"))

        self.modelLocationLbl = tk.Label(tabs.tab2, text="Model save location", font="Helvetica 10 bold")
        self.modelLocation = tk.Entry(tabs.tab2, width=30)
        self.modelButton = ttk.Button(tabs.tab2, text="Browse",
                                      command=partial(self.browseFolder, "saveDirectory"))

        self.epochLabel = tk.Label(tabs.tab2, text="Epochs: ", font="Helvetica 10 bold")
        self.epochInput = tk.Entry(tabs.tab2, width=3)
        self.progressbar = ttk.Progressbar(tabs.tab2, variable=self.progressValue, maximum=100, length=180)
        self.trainButton = ttk.Button(tabs.tab2, text="Train", command=self.runThread)

        ################################################################################################################
        # Text tags for colourful formatting
        ################################################################################################################
        self.outputText.tag_configure("info", foreground="blue")
        self.outputText.tag_configure("error", foreground="red")
        self.outputText.tag_configure("warn", foreground="yellow")
        self.outputText.tag_configure("debug", foreground="purple")

    def placeWidgets(self):
        self.outputWindowLabel.place(x=65, y=10, anchor=tk.CENTER)
        self.outputText.place(x=10, y=30)
        self.accuracyLabel.place(x=10, y=370)
        self.precisionLabel.place(x=10, y=400)
        self.precisionTxt.place(x=10, y=430)
        self.recallLabel.place(x=10 + self.placement, y=400)
        self.recallTxt.place(x=10 + self.placement, y=430)
        self.F1Label.place(x=10 + self.placement * 2, y=400)
        self.F1Txt.place(x=10 + self.placement * 2, y=430)

        # data input for training
        self.cnnFrame.place(x=600, y=20)
        self.imageInputLabel.place(x=self.rootWindow.centerTrain, y=50, anchor=tk.CENTER)
        self.imageInputText.place(x=self.rootWindow.centerTrain, y=90, anchor=tk.CENTER)
        self.imageFolderButton.place(x=self.rootWindow.centerTrain, y=130, anchor=tk.CENTER)
        self.modelLocationLbl.place(x=self.rootWindow.centerTrain, y=170, anchor=tk.CENTER)
        self.modelLocation.place(x=self.rootWindow.centerTrain, y=210, anchor=tk.CENTER)
        self.modelButton.place(x=self.rootWindow.centerTrain, y=250, anchor=tk.CENTER)
        self.epochLabel.place(x=self.rootWindow.centerTrain, y=290, anchor=tk.E)
        self.epochInput.place(x=self.rootWindow.centerTrain, y=290, anchor=tk.W)
        self.progressbar.place(x=self.rootWindow.centerTrain, y=320, anchor=tk.CENTER)
        self.trainButton.place(x=self.rootWindow.centerTrain, y=350, anchor=tk.CENTER)

    def logText(self, log_type, msg):
        self.outputText.config(state=tk.NORMAL)
        if log_type == 1:
            self.outputText.insert(tk.END, "INFO: ", "info")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 2:
            self.outputText.insert(tk.END, "WARN: ", "warn")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 3:
            self.outputText.insert(tk.END, "ERROR: ", "error")
            self.outputText.insert(tk.END, msg + "\n")
        elif log_type == 4:
            self.outputText.insert(tk.END, "DEBUG: ", "debug")
            self.outputText.insert(tk.END, msg + "\n")
        else:
            self.outputText.insert(tk.END, msg + "\n")

        self.outputText.yview(tk.END)

    def browseFolder(self, browse_type):
        if browse_type == "directory":
            self.saveLocation.set(filedialog.askdirectory())
            self.imageInputText.insert(tk.END, self.saveLocation.get())
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                    "Updating image directory to {}".format(self.saveLocation.get()))
        if browse_type == "save":
            self.saveLocation.set(tk.filedialog.askdirectory())
        elif browse_type == "saveDirectory":
            self.modelSaveLocation.set(tk.filedialog.askdirectory())
            self.modelLocation.insert(tk.END, self.modelSaveLocation.get())
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                    "Setting save location to {} for the model".format(
                                                        self.modelSaveLocation.get()))

    def runThread(self):
        # Don't block main thread with heavy I/O intensive tasks
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.info.value, "Running  training on new thread")

        if not os.path.isdir(self.saveLocation.get() + "/Train"):
            self.logText(LogInfo.error.value, "Couldn't find a Train directory within data dir")
        elif not os.path.isdir(self.saveLocation.get() + "/Validate"):
            self.logText(LogInfo.error.value, "Couldn't find a Validate directory within data dir")
        elif not os.path.isdir(self.saveLocation.get() + "/Test"):
            self.logText(LogInfo.error.value, "Couldn't find a Test directory within data dir")
        elif not os.path.isdir(self.modelSaveLocation.get()):
            self.logText(LogInfo.error.value, "Couldn't find directory to save model to")
        else:
            # See if the csv file exists, if it does, we can use it for concatenating with the
            # cnn. The CNN alone is currently has a F1score of 94% on sign images
            if list(pathlib.Path(self.saveLocation.get() + "/Train/").rglob("*.csv")):
                if self.rootWindow.debug.get():
                    self.rootWindow.debugWindow.logText(LogInfo.info.value, "Found a CSV File")
                self.csvFound = True
            try:
                int(self.epochInput.get())
            except ValueError:
                self.logText(LogInfo.error.value, "Enter a positive digit for epochs")
            else:
                self.logText(LogInfo.info.value, "Starting training")
                self.epochs = int(self.epochInput.get())
                t1 = Thread(target=self.runTraining)
                t1.start()

    def runTraining(self):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating Xception based model")

        self.model = ourModels.CustomModel(root_window=self.rootWindow)
        #self.model.createModel()

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Starting training on the new thread")

        self.model.train()
