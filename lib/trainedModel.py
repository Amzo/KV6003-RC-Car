
#!/usr/bin/env python3

from tflite_runtime.interpreter import Interpreter
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical

import numpy as np
import os

def load_model(path, file):
	interpreter = Interpreter(os.path.join(path, file))
	interpreter.allocate_tensors()

	return interpreter


def get_prediction(model, image, distance):
	encoder = LabelEncoder()
	encoder.classes_ = np.load('models/classes.npy')

	#LabelEncoder()

	inputDetails = model.get_input_details()
	outputDetails = model.get_output_details()

	image = image / 255
	distance = distance / 100
	image = np.expand_dims(image, axis=-1)

	image = np.swapaxes(image, 0, 2)
	#img = np.append(image, distance)
	#print(np.float32(img))

	#print(img.shape)
	model.set_tensor(inputDetails[0]['index'], np.float32(image))
	model.invoke()

	result = model.get_tensor(outputDetails[0]['index'])

	result =  np.round(result)

	predString = int

	a = np.array([1,0,0,0])
	w = np.array([0,0,0,1])
	d = np.array([0,1,0,0])
	s = np.array([0,0,1,0])

	if np.array_equal(result[0],a):
		print("Moving Left")
		return "a"
	elif np.array_equal(result[0], w):
		print("Moving forward")
		return "w"
	elif np.array_equal(result[0], d):
		print("moving right")
		return "d"
	elif np.array_equal(result[0], s):
		print("Moving back")
		return "s"

#	result = result.reshape(-1, 1)
	#Y = to_categorical(result)
#	return result #encoder.inverse_transform(result)
