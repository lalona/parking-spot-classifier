from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.optimizers import Adam, SGD
import numpy as np
import os
import argparse
import json
from preprocess_data import getSubsetsSpaces, getSubsets, getNormalizedX, getFeatures
from keras.callbacks import CSVLogger, Callback, LearningRateScheduler
from datetime import datetime
import matplotlib.pyplot as plt
import math
import keras.backend as K
import csv
from compare_images_methods2 import getMethods
from contextlib import redirect_stdout

INIT_LR = 2e-4
EPOCHS = 5
BATCH_SIZE = 100

learning_rates = []
class PrintLR(Callback):
		def on_epoch_end(self, epoch, logs=None):
				learning_rates.append(K.eval(self.model.optimizer.lr))


class LossHistory(Callback):
		def on_train_begin(self, logs={}):
				self.losses = []
				self.lr = []

		def on_epoch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
				self.lr.append(step_decay(len(self.losses)))


def step_decay(epoch):
		initial_lrate = INIT_LR
		drop = 0.2
		epochs_drop = 11.0
		lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
		print('lr: {}'.format(lrate))
		return lrate




def getModel(input_shape, classes=2):
		print(input_shape)
		model = models.Sequential()
		# Input - Layer
		#model.add(layers.Dense(30, activation="relu", input_shape=input_shape))
		# Hidden - Layers
		#model.add(layers.Dense(80, activation="relu", input_shape=input_shape))
		#model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
		model.add(layers.Dense(100, activation="relu", input_shape=input_shape))
		#model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
		#model.add(layers.Dense(40, activation="relu", input_shape=input_shape))
		#model.add(layers.Dense(60, activation="relu", input_shape=input_shape))
		#model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
		# Output- Layer
		model.add(layers.Dense(classes))
		model.add(layers.Activation("softmax"))
		return model


def main():
	parser = argparse.ArgumentParser(description='Select the type of reduced.')
	parser.add_argument("-d", "--dataset-unprocess", type=str, required=True,
												help='Path to the dataset unprocessed.')
	parser.add_argument("-b", "--brightness", type=bool, required=False, default=True,
											help='Include brightness.')

	args = vars(parser.parse_args())

	dataset_unprocess_path = args["dataset_unprocess"]
	brightness = args["brightness"]
	brightness = False
	dataset_unprocess_name = os.path.splitext(dataset_unprocess_path)[0]
	database = dataset_unprocess_name.split('_')[1]
	if brightness:
		dataset_name = 'dataset_{}_brigthness'.format(database)
	else:
		dataset_name = 'dataset_{}'.format(database)
	methods = getMethods()
	for method in methods:
			dataset_name += '_{}'.format(method['name'])
	dataset_unprocess_name = dataset_name

	train_results_direcotry = os.path.join('pruebas', 'neuralnet_{}'.format(dataset_unprocess_name))
	if not os.path.isdir(train_results_direcotry):
		os.mkdir(train_results_direcotry)
	train_results_direcotry = os.path.join(train_results_direcotry, datetime.now().strftime("%d-%m-%Y_%H-%M"))
	os.mkdir(train_results_direcotry)

	plot_path = os.path.join(train_results_direcotry, 'plot.png')
	model_path = os.path.join(train_results_direcotry, 'parking_classification.model')
	csv_log_path = os.path.join(train_results_direcotry, 'epochs_log.csv')
	csv_lr_path = os.path.join(train_results_direcotry, 'learning_rates.csv')
	model_summary_path = os.path.join(train_results_direcotry, 'modelsummary.txt')


	with open(dataset_unprocess_path) as f:
			dataset_unprocess = json.load(f)

	subset_spaces = getSubsetsSpaces(dataset_unprocess)
	dataset_train, dataset_test = getSubsets(dataset_unprocess, subset_spaces, to_categorical_flag=True, brightness_feature=brightness)

	X = dataset_train['X']
	X_test = dataset_test['X']
	y = dataset_train['Y']
	y_test = dataset_test['Y']

	print(len(X))
	print(len(y))
	print(len(X_test))
	print(len(y_test))

	X = getNormalizedX(X)
	X_test = getNormalizedX(X_test)

	STEP_SIZE_TRAIN = len(X) / BATCH_SIZE
	STEP_SIZE_VALID = len(X_test) / BATCH_SIZE

	X = np.array(X).astype("float32")
	X_test = np.array(X_test).astype("float32")

	y = np.array(y)
	y_test = np.array(y_test)

	model = getModel(X[0].shape)
	# compiling the model
	opt = Adam(lr=INIT_LR)
	model.compile(
			optimizer=opt,
			loss="binary_crossentropy",
			metrics=["accuracy"]
	)
	history = LossHistory()
	lrate = LearningRateScheduler(step_decay)
	printlr = PrintLR()
	callbacks = [CSVLogger(filename=csv_log_path, separator=',', append=False), printlr]
	H = model.fit(
			X, y,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			validation_data=(X_test, y_test),
			callbacks=callbacks
	)

	print("[INFO] serializing network...")
	model.save(model_path)

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	print(H.history)
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Ocuppied/Empty")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_path)

	with open(csv_lr_path, 'w', newline='') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(learning_rates)


if __name__ == "__main__":
	main()