
#!/usr/bin/env python3

from tflite_runtime.interpreter import Interpreter
import numpy as np
import os

def load_model(path, file):
	interpreter = Interpreter(os.path.join(path, file))
	interpreter.allocate_tensors()

	return interpreter


def get_prediction(model, image, distance):
	inputDetails = model.get_input_details()
	outputDetails = model.get_output_details()

	print(inputDetails[0]['index'])
	image = image / 255
	distance = distance / 100
	image = np.expand_dims(image, axis=-1)
	print(image.shape)

	img = np.append(image, distance)
	print(np.float32(img))

	model.set_tensor(inputDetails[0]['index'], np.float32(img))
	model.invoke()

	result = model.get_tensor(outputDetails[0]['index'])

	print(result)

	return result
