#!/usr/bin/env python3
import configparser
import os

import PIL
import numpy as np
from PIL import ImageDraw, Image, ImageFont

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.prediction.models import CustomVisionErrorException
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

            self.colour.append(color)
            self.box_list.append(((left, top), (left + width, top), (left + width, top + height), (left, top + height),
                                  (left, top)))
            self.tag_name.append(prediction.tag_name)

    def filterResults(self, test_img):
        try:
            # erase previous bounding boxes
            self.box_list = []
            self.colour = []
            self.tag_name = []

            for prediction in self.results.predictions:
                if (prediction.probability * 100) > 70:
                    # bounding box
                    self.getBoundingBox(prediction, test_img)

            if len(self.box_list) > 0:
                draw = ImageDraw.Draw(test_img)
                lineWidth = int(np.array(test_img).shape[1] / 100)
                fnt = ImageFont.truetype("./data/arial.ttf", 12)

                for x in range(0, len(self.box_list)):
                    draw.line(self.box_list[x], fill=self.colour[x], width=lineWidth)
                    draw.text(self.box_list[x][0], self.tag_name[x] + ": {0:.2f}%".format(prediction.probability * 100),
                              font=fnt, fill=self.colour[x])

                    # ensure previous one has been loaded and removed
                if not os.path.exists('imageBox.jpg'):
                    test_img.save('imageBox.jpg')

            for prediction in self.results.predictions:
                # commands to send to raspberry pi on detection
                if prediction.tag_name == "car":
                    return self.box_list, str.encode("c")
                elif prediction.tag_name == "Left":
                    return self.box_list, str.encode("l")
                elif prediction.tag_name == "Right":
                    return self.box_list, str.encode("r")
                elif prediction.tag_name == "person":
                    return self.box_list, str.encode("p")
                elif prediction.tag_name == "stop":
                    return self.box_list, str.encode("t")
        except TypeError:
            return str.encode("z")
        else:
            # fall back, no object detection yet
            return str.encode("z")
