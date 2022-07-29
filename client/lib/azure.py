#!/usr/bin/env python3
import configparser
import os
import pickle
from pathlib import Path

import PIL
import numpy as np
from PIL import ImageDraw, Image, ImageFont

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.prediction.models import CustomVisionErrorException
from filelock import FileLock
from matplotlib import pyplot as plt
from msrest.authentication import ApiKeyCredentials


class CarObjectDetection:
    def __init__(self):
        self.tag_name = None
        self.results = None
        self.results_list = str
        self.box_list = []
        self.colour = []
        self.config = configparser.ConfigParser()
        self.config.read('../config/config.ini')
        self.checkImage = None

        self.__credentials = ApiKeyCredentials(in_headers={"Prediction-key": ""})
        self.__predictor = CustomVisionPredictionClient(endpoint='https://signdetection-prediction.cognitiveservices'
                                                                 '.azure.com/',
                                                        credentials=self.__credentials)

    def getPrediction(self):
        if os.path.exists('image1.jpg'):
            with open('image1.jpg', mode="rb") as test_data:
                try:
                    self.results = self.__predictor.detect_image('85177bf1-b325-4299-868e-e45f80a62bc4',
                                                                 'Iteration7',
                                                                 test_data)
                except CustomVisionErrorException:
                    # bad image stream
                    # skip this error
                    pass

    def getBoundingBox(self, prediction, test_img):
        test_img_h, test_img_w, test_img_ch = np.array(test_img).shape

        color = 'white'
        object_colors = {
            "car": "lightgreen",
            "Left": "yellow",
            "Right": "orange",
            "stop": "red",
            "person": "blue"
        }
        if prediction.tag_name in object_colors:
            color = object_colors[prediction.tag_name]
            left = prediction.bounding_box.left * test_img_w
            top = prediction.bounding_box.top * test_img_h
            height = prediction.bounding_box.height * test_img_h
            width = prediction.bounding_box.width * test_img_w

            self.box_list.append(((left, top), (left + width, top), (left + width, top + height), (left, top + height),
                                  (left, top), color, prediction.tag_name, prediction.probability))

    def filterResults(self, test_img):
        try:
            # erase previous bounding boxes
            self.box_list = []

            for prediction in self.results.predictions:
                if (prediction.probability * 100) > 70:
                    # bounding box
                    self.getBoundingBox(prediction, test_img)

            # shared memory in python can be os dependant, I can't be certain shared objects between memory
            # will work on my Windows laptop for demonstration as all testing is done on My linux machine.
            # to share a list between processes, as multiprocessing wasn't in the original design
            # pickle it and reload it in the other process at the cost of overhead

            if not os.path.exists('filelock'):
                # lock the file
                Path('filelock').touch()
                print('file is locked. dumping lists (azure process)')
                with open('box.pkl', 'wb') as file:
                    pickle.dump(self.box_list, file)
                os.remove('filelock')
                print('file is unlocked azure process')

            for prediction in self.results.predictions:
                # commands to send to raspberry pi on detection
                if prediction.tag_name == "car":
                    return str.encode("c")
                elif prediction.tag_name == "Left":
                    return str.encode("l")
                elif prediction.tag_name == "Right":
                    return str.encode("r")
                elif prediction.tag_name == "person":
                    return str.encode("p")
                elif prediction.tag_name == "stop":
                    return str.encode("t")
        except OSError as exc:
            print(exc)
