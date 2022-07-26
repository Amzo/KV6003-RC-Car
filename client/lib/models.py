import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageOps
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Lambda, Conv2D, Flatten, Dense, Dropout
from keras.layers.preprocessing.image_preprocessing import RandomBrightness, RandomFlip

from lib.debug import LogInfo


class CustomModel(tf.keras.callbacks.Callback):
    def __init__(self, root_window=None):
        self.results = None
        self.xTest = None
        self.xTrain = None
        self.model = None
        self.prediction = multiprocessing.Value('b', False)
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

    def makePrediction(self, check_frame=None):
        class_names = ['a', 'd', 'w']
        if self.rootWindow.predictTab.selectedModel.get() == "CNN":
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Getting CNN Prediction")

            predFrame = ImageOps.grayscale(check_frame)
            img_array = tf.keras.utils.img_to_array(predFrame)
            img_array = tf.expand_dims(img_array, 0)
            prediction = self.model.predict(img_array)

            return class_names[np.argmax(prediction)]

    def train(self):
        self.xTrain = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Train/",
            seed=743,
            color_mode="grayscale",
            label_mode='categorical',
            shuffle=True,
            image_size=(240, 320),
            batch_size=128)

        self.xTest = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Test/",
            color_mode="grayscale",
            label_mode='categorical',
            image_size=(240, 320),
            batch_size=128)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.rootWindow.trainTab.modelSaveLocation.get())
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0010) #, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.rootWindow.trainTab.modelSaveLocation.get() + '/' + 'model.h5',
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')

        print("creating model")
        self.createModel()

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Compiling the model loss: categorical crossentropy, optimizer: Adam")

        print("Compiling Model")
        # loss='mean_squared_error'
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        print(self.model.summary())
        print("Fitting model")
        history = self.model.fit(self.xTrain, verbose=1,
                                 epochs=self.rootWindow.trainTab.epochs,
                                 shuffle=True,
                                 batch_size=None,
                                 validation_data=self.xTest,
                                 callbacks=[self, reduce_lr, model_checkpoint_callback, tensorboard_callback], )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.rootWindow.trainTab.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        # load best model
        self.model = tf.keras.models.load_model(self.rootWindow.trainTab.modelSaveLocation.get() + '/' + 'model.h5')

        score = self.model.evaluate(self.xTest)

        precision = score[2]
        recall = score[3]
        try:
            f1Score = round(2 * ((precision * recall) / (precision + recall)), 4)
        except ZeroDivisionError:
            # something got an accuracy of 0
            f1Score = 0

        self.rootWindow.trainTab.precision.set(round(precision, 2))
        self.rootWindow.trainTab.recall.set(round(recall, 2))
        self.rootWindow.trainTab.f1.set(round(f1Score, 2))

    def loadModel(self):
        self.model = tf.keras.models.load_model(f'{self.rootWindow.predictTab.modelLocation.get()}/model.h5')

    def createModel(self):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating a model for distance and previous key")

        input_image = Input(shape=(240, 320, 1), name='input')

        x = input_image
        x = tf.keras.layers.Cropping2D(cropping=((90, 0), (0, 0)))(x)
        x = RandomBrightness(factor=0.2)(x)
        x = RandomFlip(mode="vertical")(x)

        x = Lambda(lambda x: x / 255.0, input_shape=(240, 320, 1))(x)

        x = Conv2D(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(48, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Flatten(name='flat')(x)

        x = Dense(1164, name='dense11')(x)
        x = Dropout(0.5, name='dropout11')(x)
        x = Dense(100, name='dense1', activation='relu')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(50, name='dense2')(x)
        x = Dropout(0.5, name='dropout2')(x)
        x = Dense(25, name='dense3')(x)
        x = tf.keras.layers.GaussianDropout(0.4, name='dropout3')(x)
        x = Dense(25, name='dense4', activation='relu')(x)

        steering_output = Dense(3, activation='softmax', name="steering")(x)

        self.model = Model(inputs=[input_image], outputs=[steering_output])
