#!/usr/bin/env python3
import configparser
import os

import cv2
import numpy as np
from PIL.ImageDraw import ImageDraw
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials


class CarObjectDetection:
    def __init__(self):
        self.results = None
        self.results_list = None
        self.config = configparser.ConfigParser()
        self.config.read('../config/config.ini')
        self.checkImage = None

        self.__credentials = ApiKeyCredentials(in_headers={"Prediction-key": "24072118e84c4da88f7f4c87c93c2317"})
        self.__predictor = CustomVisionPredictionClient(endpoint='https://signdetection-prediction.cognitiveservices.azure.com/',
                                                        credentials=self.__credentials)

    def getPrediction(self):
        if os.path.exists('image.jpg'):
            with open('image.jpg', mode="rb") as test_data:
                self.results = self.__predictor.detect_image('85177bf1-b325-4299-868e-e45f80a62bc4',
                                                             'Iteration5',
                                                             test_data)

    def filterResults(self):
        self.results_list = []
        try:
            for prediction in self.results.predictions:
                if (prediction.probability * 100) > 70:
                    if prediction.tag_name == "car":
                        self.results_list.append("c")
                    elif prediction.tag_name == "Left":
                        self.results_list.append("l")
                    elif prediction.tag_name == "Right":
                        self.results_list.append("r")
                    elif prediction.tag_name == "person":
                        self.results_list.append("p")
                    elif prediction.tag_name == "stop":
                        self.results_list.append("t")
        except AttributeError:
            pass

