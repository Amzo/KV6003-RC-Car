#!/usr/bin/env python3
import os

from lib import gui


def cleanup():
    if os.path.exists('filelock'):
        os.remove('filelock')
    if os.path.exists('box.pkl'):
        os.remove('box.pkl')
    if os.path.exists('image.jpg'):
        os.remove('image.jpg')

# make sure no files remain from previous run if it ended without cleaning
cleanup()

ourGUI = gui.Gui()
# ourGUI.updateWindow()
threadRunning = True
