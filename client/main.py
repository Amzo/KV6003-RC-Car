#!/usr/bin/env python3
import os

from lib import gui


def cleanup():
    if os.path.exists('filelock'):
        os.remove('filelock')


# make sure no files remain from previous run if it ended without cleaning
cleanup()

ourGUI = gui.Gui()
# ourGUI.updateWindow()
threadRunning = True
