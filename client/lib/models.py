import multiprocessing
from tkinter import messagebox

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import ImageOps
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Lambda, Flatten, Dense
from keras.layers.preprocessing.image_preprocessing import RandomBrightness, RandomFlip
from lib.debug import LogInfo
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.vis_utils import plot_model


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

    def show_training_hist(self, history):
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

    def makePrediction(self, check_frame=None):
        class_names = ['a', 'd', 't', 'w']
        if self.rootWindow.predictTab.selectedModel.get() == "CNN":
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Getting CNN Prediction")

            #predFrame = ImageOps.grayscale(check_frame)
            img_array = tf.keras.utils.img_to_array(check_frame)
            img_array = tf.expand_dims(img_array, 0)
            prediction = self.model.predict(img_array)

            return class_names[np.argmax(prediction)]

    def show_classification_report(self):
        y_true = []
        y_pred = []

        for x, y in self.xTest:
            y = tf.argmax(y, axis=1)
            y_true.append(y)
            y_pred.append(tf.argmax(self.model.predict(x), axis=1))

        y_pred = tf.concat(y_pred, axis=0)
        y_true = tf.concat(y_true, axis=0)

        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(cm / np.sum(cm), annot=True,
                    fmt='.2%', cmap='Blues')
        ax1.set_ylabel('True Values', fontsize=14)
        ax1.set_xlabel('Predicted Values', fontsize=14)
        ax1.xaxis.set_ticklabels(['a', 'd', 't', 'w'])
        ax1.yaxis.set_ticklabels(['a', 'd', 't', 'w'])
        plt.show()

        target_names = ['a', 'd', 't', 'w']

        print(classification_report(y_true, y_pred, target_names=target_names))

    def train(self):
        from neptune.new.integrations.tensorflow_keras import NeptuneCallback

        self.xTrain = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Train/",
            seed=743,
            color_mode="rgb",
            label_mode='categorical',
            shuffle=True,
            image_size=(240, 320),
            batch_size=16)

        self.xTest = tf.keras.utils.image_dataset_from_directory(
            self.rootWindow.trainTab.saveLocation.get() + "/Test/",
            color_mode="rgb",
            label_mode='categorical',
            image_size=(240, 320),
            batch_size=16)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.rootWindow.trainTab.modelSaveLocation.get())
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0010)  # , epsilon=1e-8, beta_1=0.9, beta_2=0.999)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.rootWindow.trainTab.modelSaveLocation.get() + '/' + 'model.h5',
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')

        print("creating model")

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Compiling the model loss: categorical crossentropy, optimizer: Adam")

        run = neptune.init(
            project="amzo1337/RC-CAR",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYWE5ZDk4MC00N2RmLTRmZjYtYjEzYS0zMjdjNDNjMWJkYzEifQ==",
        )

        params = {"lr": 0.0010, "epochs": self.rootWindow.trainTab.epochs, "batch_size": 16}

        run["parameters"] = params

        neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

        for i in range(2):
            if i == 0:
                print("Creating transfer learning model")
                self.createModel(i)
            else:
                run = neptune.init(
                    project="amzo1337/RC-CAR",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYWE5ZDk4MC00N2RmLTRmZjYtYjEzYS0zMjdjNDNjMWJkYzEifQ==",
                )
                params = {"lr": 1e-5, "epochs": self.rootWindow.trainTab.epochs, "batch_size": 16}
                print("Creating fine tuning model")
                run["parameters"] = params
                neptune_cbk = NeptuneCallback(run=run, base_namespace="training")
                self.createModel(i)
                opt = tf.keras.optimizers.Adam(params["lr"])

            self.rootWindow.trainTab.logText(LogInfo.info.value, "Compiling Model")
            # loss='mean_squared_error'
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=opt,
                               metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            print(self.model.summary())
            self.rootWindow.trainTab.logText(LogInfo.info.value, "Fitting model")

            history = self.model.fit(self.xTrain, verbose=1,
                                     epochs=self.rootWindow.trainTab.epochs,
                                     shuffle=True,
                                     batch_size=None,
                                     validation_data=self.xTest,
                                     callbacks=[self, reduce_lr, model_checkpoint_callback, tensorboard_callback, neptune_cbk], )

            self.show_training_hist(history)

            # load best model
            self.model = tf.keras.models.load_model(self.rootWindow.trainTab.modelSaveLocation.get() + '/' + 'model.h5')
            score = self.model.evaluate(self.xTest)

            for j, metric in enumerate(score):
                run["eval/{}".format(self.model.metrics_names[j])] = metric

            run.stop()

            self.show_classification_report()

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
        try:
            self.model = tf.keras.models.load_model(f'{self.rootWindow.predictTab.modelLocation.get()}/model.h5')
        except OSError:
            messagebox.showerror("Error", "Couldn't find a model ending in specified directory")

    def createModel(self, i):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating a model for distance and previous key")

        input_image = Input(shape=(240, 320, 3), name='input')

        self.base_model = tf.keras.applications.Xception(
            include_top=False,
            input_shape=(150, 320, 3),
            weights="imagenet",
            pooling='max')

        if i == 0:
            self.base_model.trainable = False
        elif i == 1:
            print("setting trainable to True")
            self.base_model.trainable = True

        x = input_image

        x = tf.keras.layers.Cropping2D(cropping=((90, 0), (0, 0)))(x)
        x = RandomBrightness(factor=0.2)(x)
        x = RandomFlip(mode="vertical")(x)

        x = Lambda(lambda x: x / 255.0, input_shape=(240, 320, 1))(x)

        x = self.base_model(x, training=False)

        x = Flatten(name='flat')(x)

        x = Dense(150, name='dense1', activation='relu')(x)
        x = Dense(50, name='dense3')(x)
        x = tf.keras.layers.GaussianDropout(0.4, name='dropout3')(x)
        x = Dense(50, name='dense4', activation='relu')(x)

        steering_output = Dense(4, activation='softmax', name="steering")(x)

        self.model = Model(inputs=[input_image], outputs=[steering_output])
