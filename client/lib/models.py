import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import concatenate
from tensorflow import train

from sklearn.utils import shuffle

from lib.debug import LogInfo
from lib import images


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def crop_bottom_half(image):
    height, width, channels = image.shape
    croppedImage = image[int(height / 2):height, 0:width]
    return croppedImage


def appendImage(dirs=None, index=None, img_file=None):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    assert img.shape == (100, 100, 3)


class customModel(tf.keras.callbacks.Callback):
    def __init__(self, root_window=None):
        self.prevKey = None
        self.carDataSet = None
        self.encoder = None
        self.results = None
        self.prevF1Score = 0
        self.Fold_Test_Input2 = None
        self.Fold_Test_OutPut = None
        self.Fold_Test_Input1 = None
        self.Fold_Train_OutPut = None
        self.Fold_Train_Input2 = None
        self.Fold_Train_Input1 = None
        self.fold_var = 1
        self.x2Test = None
        self.x1Test = None
        self.yTest = None
        self.yValidate = None
        self.yTrain = None
        self.x2Validate = None
        self.x1Validate = None
        self.x2Train = None
        self.x1Train = None
        self.testLabel = None
        self.validateLabel = None
        self.trainLabel = None
        self.signLanguageSet = None
        self.modelFeatures = None
        self.model = None
        self.signFeatures = None
        self.testImages = None
        self.testFeatures = None
        self.encoded_test_Y = None
        self.encoded_Y = None
        self.imgFeatures = None
        self.train_image = []
        self.train_label = []
        self.trainIndices = []

        self.validate_image = []
        self.validate_label = []
        self.validateIndices = []

        self.test_image = []
        self.test_label = []
        self.testIndices = []
        self.rootWindow = root_window

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        updateValue = 100.00 / int(self.rootWindow.trainTab.epochInput.get())

        self.rootWindow.trainTab.logText(LogInfo.info.value,
                                         "End epoch {}; loss: {} - accuracy: {}"
                                         .format(epoch, round(logs['loss'], 4), round(logs['accuracy'], 4), ))

        # epoch starts from 0, add + 1
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Updating progress bar by (epoch {} * {})".format(epoch, updateValue))

        self.rootWindow.trainTab.progressValue.set(epoch * updateValue)

    def makePrediction(self, points=None, check_frame=None):
        if self.rootWindow.predictTab.selectedModel.get() == "CNN":
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Getting CNN Prediction")

            imgRez = cv2.resize(check_frame, (200, 100))
            imgRez = imgRez / 255
            image = crop_bottom_half(imgRez)
            check_frame = np.array(image)

            predFrame = np.expand_dims(check_frame, axis=0)

            if self.prevKey is None:
                self.prevKey = np.array([0.33])

            prediction = self.model.predict([predFrame, self.prevKey])
            self.results = self.encoder.inverse_transform(prediction)

            if self.results == "w":
                self.prevKey = np.array([0.33])
            elif self.results == "d":
                self.prevKey = np.array([0.66])
            else:
                self.prevKey = np.array([0.99])

    def train(self):
        self.x1Train = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Train/",
            validation_split=0.33,
            subset="training",
            seed=743,
            color_mode="grayscale",
            label_mode='categorical',
            image_size=(240, 320),
            batch_size=128)

        self.x1Val = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Train/",
            validation_split=0.33,
            subset="validation",
            seed=742,
            color_mode="grayscale",
            label_mode='categorical',
            image_size=(240, 320),
            batch_size=128)

        # print("creating encoder")

        # encoder = LabelBinarizer()
        # encoder.fit(trainLabels)
        # self.yTrain = encoder.transform(trainLabels)

        # output = open(self.rootWindow.trainTab.modelSaveLocation.get() + '/classes.pkl', 'wb')
        # pickle.dump(encoder, output)
        # output.close()

        # self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} training images".format(len(self.x1Train)))

        # prepare to use kfold on multiple inputs
        # kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.rootWindow.trainTab.modelSaveLocation.get())

        average_scores = []

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

        opt = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, decay=0.01)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.rootWindow.trainTab.modelSaveLocation.get() + '/' + get_model_name(self.fold_var),
            monitor='accuracy', verbose=1,
            save_best_only=True, mode='max')

        print("creating model")
        self.createModel()

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Compiling the model loss: categorical crossentropy, optimizer: Adam")

        print("Compiling Model")
        # loss='mean_squared_error'
        self.model.compile(loss='mean_squared_error', optimizer=opt,
                           metrics=['accuracy']) #, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        self.rootWindow.trainTab.logText(LogInfo.info.value,
                                         "Starting training on fold number {}".format(self.fold_var))

        print(self.model.summary())
        print("Fitting model")
        self.model.fit(self.x1Train, verbose=1,
                       epochs=self.rootWindow.trainTab.epochs,
                       shuffle=True,
                       batch_size=None,
                       validation_data=self.x1Val,
                       callbacks=[self, reduce_lr, model_checkpoint_callback, tensorboard_callback], )

        class_names = self.x1Train.class_names
        print(class_names)
        for image, labels in self.x1Train.take(1):
            for i in range(9):
                img = np.expand_dims(image[i], 0)
                print(labels[i][:].numpy())
                predictions = self.model.predict(img)
                score = tf.nn.softmax(predictions[0])
                print(score)
                print(
                    "This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(class_names[np.argmax(score)], 100 * np.max(score))
                )



        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Loading best saved model to run on test set")

        self.model = tf.keras.models.load_model(
            self.rootWindow.trainTab.modelSaveLocation.get() + "/model_" + str(self.fold_var) + ".h5")

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Evaluating the best model on test data")

        score = self.model.evaluate(self.x1Val)

        print("precision")
        precision = score[2]

        print("recall")
        recall = score[3]

        print("calculations")
        try:
            f1Score = round(2 * ((precision * recall) / (precision + recall)), 4)
        except ZeroDivisionError:
            # something got an accuracy of 0
            f1score = 0

        # if f1Score > self.prevF1Score:
        #        self.model.save(self.rootWindow.trainTab.modelLocation.get() + "/bestModel")

        print("updating windows")
        self.rootWindow.trainTab.precision.set(round(precision, 2))
        self.rootWindow.trainTab.recall.set(round(recall, 2))
        self.rootWindow.trainTab.f1.set(round(f1Score, 2))

    def loadModel(self):
        self.encoder = pickle.load(open(self.rootWindow.predictTab.modelLocation.get() + "/../classes.pkl", 'rb'))
        self.model = tf.keras.models.load_model(self.rootWindow.predictTab.modelLocation.get())

    def createModel(self):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating a model for distance and previous key")

        input_image = Input(shape=(240, 320, 1), name='input')

        x = input_image

        x = Lambda(lambda x: x / 255.0, input_shape=(240, 320, 1))(x)
        # x = Cropping2D(cropping=((90, 0), (0, 0)))(x)

        x = Conv2D(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Flatten(name='flat')(x)

        x = Dense(1164, name='dense11')(x)
        x = Dropout(0.1, name='dropout11')(x)
        x = Dense(100, name='dense1')(x)
        x = Dropout(0.1, name='dropout1')(x)
        x = Dense(50, name='dense2')(x)
        x = Dropout(0.1, name='dropout2')(x)
        x = Dense(25, name='dense3')(x)
        x = Dropout(0.1, name='dropout3')(x)
        x = Dense(9, name='dense4', activation='relu')(x)

        steering_output = Dense(3, name="steering")(x)

        self.model = Model(inputs=[input_image], outputs=[steering_output])
