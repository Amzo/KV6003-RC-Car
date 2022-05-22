import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import concatenate
from sklearn.utils import shuffle

from lib.debug import LogInfo


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
                                         "End epoch {}; loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {}"
                                         .format(epoch, round(logs['loss'], 4), round(logs['accuracy'], 4),
                                                 round(logs['val_loss'], 4), round(logs['val_accuracy'], 4), ))

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

    def gen_flow_for_two_inputs(self, gen):
        genX1 = gen.flow(self.x1Train, self.yTrain, batch_size=64, seed=987654321)
        genX2 = gen.flow(self.x1Train, self.x2Train, batch_size=64, seed=987654321)

        while True:
            X1i = genX1.next()
            X2i = genX2.next()

            # Assert arrays are equal - this was for peace of mind, but slows down training
            np.testing.assert_array_equal(X1i[0], X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

    def train(self):
        self.carDataSet = pd.read_csv(self.rootWindow.trainTab.saveLocation.get() + "/Train/labels.csv",
                                      names=["Image", "prevKey", "Key"])

        updateValue = 100 / len(self.carDataSet)

        for i in range(self.carDataSet.shape[0]):
            img = cv2.imread(self.rootWindow.trainTab.saveLocation.get() + "/Train/" + self.carDataSet['Image'][i],
                             cv2.IMREAD_COLOR)

            image = crop_bottom_half(img)

            self.train_image.append(np.asarray(image))
            self.train_label.append(self.carDataSet['Key'][i])

            self.rootWindow.trainTab.progressValue.set(i + 1 * updateValue)

        # drop none needed columns and create the features and labels for training
        self.carDataSet.drop('Image', inplace=True, axis=1)
        self.carDataSet.drop('Key', inplace=True, axis=1)
        featureList = self.carDataSet.iloc[:]

        self.carDataSet = pd.read_csv(self.rootWindow.trainTab.saveLocation.get() + "/Validate/labels.csv",
                                      names=["Image", "prevKey", "Key"])

        updateValue = 100 / len(self.carDataSet)

        for i in range(self.carDataSet.shape[0]):
            img = cv2.imread(self.rootWindow.trainTab.saveLocation.get() + "/Validate/" + self.carDataSet['Image'][i],
                             cv2.IMREAD_COLOR)
            img = crop_bottom_half(img)
            # normalise data, data augmentation does this for train set after augmenting
            # though validation and test sets need to be normalised by me as augmentation is not
            # applied to them.
            img = img / 255

            self.validate_image.append(np.asarray(img))
            self.validate_label.append(self.carDataSet['Key'][i])

            self.rootWindow.trainTab.progressValue.set(i + 1 * updateValue)

        self.carDataSet.drop('Image', inplace=True, axis=1)
        self.carDataSet.drop('Key', inplace=True, axis=1)

        validateFeatureList = self.carDataSet.iloc[:]

        self.carDataSet = pd.read_csv(self.rootWindow.trainTab.saveLocation.get() + "/Test/labels.csv",
                                      names=["Image", "prevKey", "Key"])

        updateValue = 100 / len(self.carDataSet)

        for i in range(self.carDataSet.shape[0]):
            img = cv2.imread(self.rootWindow.trainTab.saveLocation.get() + "/Test/" + self.carDataSet['Image'][i],
                             cv2.IMREAD_COLOR)
            img = crop_bottom_half(img)
            img = img / 255

            self.test_image.append(np.asarray(img))
            self.test_label.append(self.carDataSet['Key'][i])

            self.rootWindow.trainTab.progressValue.set(i + 1 * updateValue)

        # drop none needed columns and create the features and labels for training
        self.carDataSet.drop('Image', inplace=True, axis=1)
        self.carDataSet.drop('Key', inplace=True, axis=1)

        testFeatureList = self.carDataSet.iloc[:]

        self.x1Train = np.array(self.train_image)
        trainLabels = np.array(self.train_label)
        self.x2Train = np.array(featureList)
        print(len(self.x2Train))

        self.x1Validate = np.array(self.validate_image)
        validateLabels = np.array(self.validate_label)
        self.x2Validate = np.array(validateFeatureList)

        self.x1Test = np.array(self.test_image)
        testLabels = np.array(self.test_label)
        self.x2Test = np.array(testFeatureList)

        encoder = LabelBinarizer()
        encoder.fit(trainLabels)
        self.yTrain = encoder.transform(trainLabels)
        self.yValidate = encoder.transform(validateLabels)
        self.yTest = encoder.transform(testLabels)

        output = open(self.rootWindow.trainTab.modelSaveLocation.get() + '/classes.pkl', 'wb')
        pickle.dump(encoder, output)
        output.close()

        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} training images".format(len(self.x1Train)))
        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} validation images".format(len(self.x1Validate)))
        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} testing images".format(len(self.x1Test)))

        self.x1Train, self.x2Train, self.yTrain = shuffle(self.x1Train, self.x2Train, self.yTrain, random_state=0)

        gen = ImageDataGenerator(
            vertical_flip=True,
            brightness_range=[0.4, 1.7],
            channel_shift_range=50,
            rescale=1. / 255.0, )

        gen_flow = self.gen_flow_for_two_inputs(gen)

        # prepare to use kfold on multiple inputs
        kfold = KFold(n_splits=3, shuffle=True)

        inputID = []
        outputID = []

        # since we're using kfold, join the validation data to train data before splitting
        # keep test data model evaluation
        np.concatenate((self.x1Train, self.x1Validate), axis=0)
        np.concatenate((self.x2Train, self.x2Validate), axis=0)
        np.concatenate((self.yTrain, self.yValidate), axis=0)

        for x in range(len(self.x1Train)):
            inputID.append(self.x1Train[x])
            outputID.append(self.yTrain[x])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.rootWindow.trainTab.modelSaveLocation.get())

        for trainID, testID in kfold.split(inputID, outputID):
            # Call checkpoint each iteration to update the model save name
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                self.rootWindow.trainTab.modelSaveLocation.get() + '/' + get_model_name(self.fold_var),
                monitor='val_accuracy', verbose=1,
                save_best_only=True, mode='max')

            self.Fold_Train_Input1, self.Fold_Train_Input2 = self.x1Train[trainID], self.x2Train[trainID]
            self.Fold_Train_OutPut = self.yTrain[trainID]

            self.Fold_Test_Input1, self.Fold_Test_Input2 = self.x1Train[testID], self.x2Train[testID]
            self.Fold_Test_OutPut = self.yTrain[testID]

            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

            if self.fold_var == 1:
                # Since the weights are loaded after every kfold has been trained, only set the optimizer once
                self.createModel()
                opt = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

                if self.rootWindow.debug.get():
                    self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                        "Compiling the model loss: categorical crossentropy, optimizer: Adam")

                self.model.compile(loss='categorical_crossentropy', optimizer=opt,
                                   metrics=['categorical_crossentropy', 'accuracy', tf.keras.metrics.Precision(),
                                            tf.keras.metrics.Recall()])

                print(self.model.summary())
            else:
                # Load best previous weights
                self.model.load_weights(
                    self.rootWindow.trainTab.modelSaveLocation.get() + "/model_" + str(self.fold_var - 1) + ".h5")

            self.rootWindow.trainTab.logText(LogInfo.info.value,
                                             "Starting training on fold number {}".format(self.fold_var))
            self.model.fit(gen_flow, verbose=1,
                           epochs=self.rootWindow.trainTab.epochs,
                           validation_data=(
                               [self.Fold_Test_Input1, self.Fold_Test_Input2], self.Fold_Test_OutPut),
                           steps_per_epoch=len(self.Fold_Train_Input1) / 64,
                           shuffle=True,
                           callbacks=[self, model_checkpoint_callback, tensorboard_callback], )

            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Loading best saved model")

            self.model = tf.keras.models.load_model(
                self.rootWindow.trainTab.modelSaveLocation.get() + "/model_" + str(self.fold_var) + ".h5")

            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Evaluating the best model on test data")

            score = self.model.evaluate([self.x1Test, self.x2Test], self.yTest)

            # Score is a list containing loss, categorical_crossentropy, accuracy, precision, recall

            precision = score[3]
            recall = score[4]
            try:
                f1Score = round(2 * ((precision * recall) / (precision + recall)), 4)
            except ZeroDivisionError:
                # something got an accuracy of 0
                f1score = 0

            if f1Score > self.prevF1Score:
                self.model.save(self.rootWindow.trainTab.modelLocation.get() + "/bestModel")

                self.rootWindow.trainTab.precision.set(round(precision, 2))
                self.rootWindow.trainTab.recall.set(round(recall, 2))
                self.rootWindow.trainTab.f1.set(round(f1Score, 2))

            self.rootWindow.trainTab.logText(LogInfo.info.value, "Finish train Fold number {} ".format(self.fold_var))
            self.fold_var += 1
            self.prevF1Score = f1Score

    def loadModel(self):
        self.encoder = pickle.load(open(self.rootWindow.predictTab.modelLocation.get() + "/../classes.pkl", 'rb'))
        self.model = tf.keras.models.load_model(self.rootWindow.predictTab.modelLocation.get())

    def createModel(self):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating a model for distance and previous key")

        distance = tf.keras.Input(shape=(1,), name='distance')
        x = tf.keras.layers.Dense(32, activation="relu", activity_regularizer=regularizers.L2(0.01))(distance)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation="relu", activity_regularizer=regularizers.L2(0.01))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.Model(inputs=distance, outputs=x)

        img = (50, 200, 3)

        inputs = tf.keras.Input(shape=img, name='xception')

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Loading xception model")

        xceptionModel = tf.keras.applications.xception.Xception(input_tensor=inputs, include_top=False,
                                                                weights='imagenet')

        for layer in xceptionModel.layers[:-2]:
            layer.trainable = False

        y = tf.keras.layers.Dense(512, activation="relu", activity_regularizer=regularizers.L2(0.01))(
            xceptionModel.output, training=True)
        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(256, activation="relu", )(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Dense(128, activation="relu")(y)
        y = tf.keras.Model(xceptionModel.input, y)

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Concatenating outputs of distance model and xception model")
        combined = concatenate([x.output, y.output])

        combined = tf.keras.layers.Dense(64, activation="relu")(combined)
        z = tf.keras.layers.Dense(3, activation="softmax")(combined)

        self.model = tf.keras.Model(inputs=[y.input, x.input], outputs=z)
