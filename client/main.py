#!/usr/bin/env python3
import os

from lib import gui


def cleanup():
    if os.path.exists('filelock'):
        os.remove('filelock')


ourGUI = gui.Gui()
# ourGUI.updateWindow()
threadRunning = True
