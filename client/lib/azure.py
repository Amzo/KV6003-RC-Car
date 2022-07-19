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

        self.__credentials = ApiKeyCredentials(in_headers={"Prediction-key": self.config['azure']['cv_key']})
        print(self.config['azure']['cv_endpoint'])
        self.__predictor = CustomVisionPredictionClient(endpoint=self.config['azure']['cv_endpoint'],
                                                        credentials=self.__credentials)

    def getPrediction(self):
        cv2.imwrite('data/image.jpg', self.checkImage)

        if os.path.exists('data/image.jpg'):
            with open('data/image.jpg', mode="rb") as test_data:
                self.results = self.__predictor.detect_image(self.config['azure']['project_id'],
                                                             self.config['azure']['model_name'],
                                                             test_data)

    def filterResults(self):
        self.results_list = []
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

        print(self.results_list)
