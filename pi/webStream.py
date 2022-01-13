#!/usr/bin/env python

import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server

with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
	camera.start_recording("../Web/video/stream.mjpeg", format='mjpeg')
	camera.wait_recording(60)
	camera.stop_recording()
