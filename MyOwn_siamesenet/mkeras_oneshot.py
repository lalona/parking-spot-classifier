from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.models import Model, Sequential
from skimage.io import imread
from skimage.transform import resize
import keras
from keras.optimizers import Adam
import numpy as np
import os
import argparse
import json
from preprocess_data import getSubsetsSpaces, getSubsets, getSubset_dummy
from datetime import datetime
import keras.backend as K
from keras.callbacks import  CSVLogger, Callback
import matplotlib.pyplot as plt
import csv

from keras.preprocessing.image import load_img, img_to_array
INIT_LR = 2e-3
EPOCHS = 3
BATCH_SIZE = 50
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_CHANNEL = 3
NUM_CLASSES = 2


learning_rates = []

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

class PrintLR(Callback):
		def on_epoch_end(self, epoch, logs=None):
				learning_rates.append(K.eval(self.model.optimizer.lr))
				print(K.eval(self.model.optimizer.lr))

def createSiameseNet(width, height, depth, classes, lr):
		input_shape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
				input_shape = (depth, height, width)

		print(input_shape)
		left_input = Input(input_shape)
		right_input = Input(input_shape)
		# build convnet to use in each siamese 'leg'
		img_input = Input(shape=input_shape)
		x = conv2d_bn(img_input, 8, 10, 10, strides=(4, 4), padding='same')
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		x = conv2d_bn(x, 10, 3, 3, padding='same')
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		x = GlobalAveragePooling2D(name='avg_pool')(x)
		x = Dropout(0.2)(x)
		#x = Flatten()(x)
		#x = Dense(48, activation='relu')(x)
		#x = Dropout(0.2)(x)
		# softmax classifier

		convnet = Model(inputs=img_input, outputs=x)
		print(convnet.summary())
		# call the convnet Sequential model on each of the input tensors so params will be shared
		encoded_l = convnet(left_input)
		encoded_r = convnet(right_input)
		# layer to merge two encoded inputs with the l1 distance between them
		L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
		# call this layer on list of two input tensors.
		L1_distance = L1_layer([encoded_l, encoded_r])
		prediction = Dense(classes, activation='softmax')(L1_distance)
		siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)


		optimizer = Adam(lr)
		# //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
		siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		return siamese_net, 'siamese_net'

# https://github.com/keras-team/keras/issues/8130
class DataGenerator(keras.utils.Sequence):
		"""Generates data for Keras."""

		def __init__(self, dataset, ave=None, std=None, batch_size=BATCH_SIZE, dim=(IMAGE_WIDTH, IMAGE_HEIGHT),
								 n_channels=3,
								 n_classes=NUM_CLASSES, shuffle=True):
				"""Initialization.

				Args:
						img_files: A list of path to image files.
						clinical_info: A dictionary of corresponding clinical variables.
						labels: A dictionary of corresponding labels.
				"""
				self.dataset = dataset
				self.batch_size = batch_size
				self.dim = dim
				if ave is None:
						self.ave = np.zeros(n_channels)
				else:
						self.ave = ave
				if std is None:
						self.std = np.zeros(n_channels) + 1
				else:
						self.std = std

				self.n_channels = n_channels
				self.n_classes = n_classes
				self.shuffle = shuffle
				self.on_epoch_end()

		def __len__(self):
				"""Denotes the number of batches per epoch."""
				return int(np.floor(len(self.dataset) / self.batch_size))

		def __getitem__(self, index):
				"""Generate one batch of data."""
				# Generate indexes of the batch
				indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

				# Find list of IDs
				img_files_temp = [self.dataset[k] for k in indexes]

				# Generate data
				X, y = self.__data_generation(img_files_temp)

				return X, y

		def on_epoch_end(self):
				"""Updates indexes after each epoch."""
				self.indexes = np.arange(len(self.dataset))
				if self.shuffle == True:
						np.random.shuffle(self.indexes)

		def __data_generation(self, img_files_temp):
				"""Generates data containing batch_size samples."""
				# X : (n_samples, *dim, n_channels)
				# X = [np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))]
				X_img = []
				X_img_compare = []
				y = np.empty((self.batch_size), dtype=int)

				# Generate data
				for i, img_files in enumerate(img_files_temp):
						img_file = img_files['X1']
						img_file_compare = img_files['X2']
						img = load_img(img_file, target_size=self.dim)
						x = img_to_array(img)
						img_compare = load_img(img_file_compare, target_size=self.dim)
						x2 = img_to_array(img_compare)

						""" 
						# Read image
						img = imread(img_file)
						# Resize
						img = resize(img, output_shape=self.dim, mode='constant', preserve_range=True)

						# Read image
						img_compare = imread(img_file_compare)
						# Resize
						img_compare = resize(img_compare, output_shape=self.dim, mode='constant', preserve_range=True)

						# Normalization
						for ch in range(self.n_channels):
								img[:, :, ch] = (img[:, :, ch] - self.ave[ch]) / self.std[ch]

						for ch2 in range(self.n_channels):
								img_compare[:, :, ch2] = (img_compare[:, :, ch2] - self.ave[ch2]) / self.std[ch2]
						"""
						X_img.append(x)
						X_img_compare.append(x2)
						y[i] = img_files['Y']
				X_img = np.array(X_img, dtype="float") / 255.0
				X_img_compare = np.array(X_img_compare, dtype="float") / 255.0
				X = [X_img, X_img_compare]
				return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def main():
	parser = argparse.ArgumentParser(description='Select the type of reduced.')
	parser.add_argument("-d", "--dataset-unprocess", type=str, required=True,
												help='Path to the dataset unprocessed.')

	args = vars(parser.parse_args())

	dataset_unprocess_path = args["dataset_unprocess"]
	dataset_unprocess_name = os.path.splitext(dataset_unprocess_path)[0]
	train_results_direcotry = os.path.join('pruebas', 'neuralnet_{}'.format(dataset_unprocess_name))
	if not os.path.isdir(train_results_direcotry):
		os.mkdir(train_results_direcotry)
	train_results_direcotry = os.path.join(train_results_direcotry, datetime.now().strftime("%d-%m-%Y_%H-%M"))
	os.mkdir(train_results_direcotry)

	plot_path = os.path.join(train_results_direcotry, 'plot.png')
	model_path = os.path.join(train_results_direcotry, 'parking_classification.model')
	csv_log_path = os.path.join(train_results_direcotry, 'epochs_log.csv')
	csv_lr_path = os.path.join(train_results_direcotry, 'learning_rates.csv')
	train_details_path = os.path.join(train_results_direcotry, 'details.json')
	dataset_used_path = os.path.join(train_results_direcotry, 'dataset_used_path.json')

	with open(dataset_unprocess_path) as f:
			dataset_unprocess = json.load(f)

	subset_spaces = getSubsetsSpaces(dataset_unprocess)
	dataset_train, dataset_test = getSubsets(dataset_unprocess, subset_spaces, to_categorical_flag=False)
	print(len(dataset_train))

	#dataset_train = getSubset_dummy(dataset_train, 60*1000)
	#dataset_test = getSubset_dummy(dataset_test, 10 * 1000)
	print(len(dataset_train))

	train_datagen = DataGenerator(dataset=dataset_train, batch_size=BATCH_SIZE, dim=(IMAGE_WIDTH, IMAGE_HEIGHT))
	val_datagen = DataGenerator(dataset=dataset_test, batch_size=BATCH_SIZE, dim=(IMAGE_WIDTH, IMAGE_HEIGHT))

	printlr = PrintLR()
	callbacks = [CSVLogger(filename=csv_log_path, separator=',', append=False), printlr]  # , history, lrate]
	model, architecture_name = createSiameseNet(IMAGE_WIDTH, IMAGE_HEIGHT, 3, NUM_CLASSES, INIT_LR)
	STEP_SIZE_TRAIN = len(dataset_train) // BATCH_SIZE
	STEP_SIZE_VALID = len(dataset_test) // BATCH_SIZE
	H = model.fit_generator(train_datagen,
															 steps_per_epoch=STEP_SIZE_TRAIN,
															 epochs=EPOCHS,
															 verbose=1,
															 validation_data=val_datagen,
															 validation_steps=STEP_SIZE_VALID,
														   callbacks= callbacks)



	# save the model to disk
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

	details = {
			'epochs': str(EPOCHS),
			'batch_size': str(BATCH_SIZE),
			'image_size': "{}x{}".format(IMAGE_WIDTH, IMAGE_HEIGHT),
			'arquitecture': architecture_name,
			'augmented_data': 'True'
	}

	dataset_used = {'base': dataset_unprocess_path, 'train': dataset_train, 'test': dataset_test}
	with open(os.path.join(dataset_used_path), 'w') as outfile:
			json.dump(dataset_used, outfile)

	with open(os.path.join(train_details_path), 'w') as outfile:
			json.dump(details, outfile)

	with open(csv_lr_path, 'w', newline='') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(learning_rates)

if __name__ == "__main__":
		main()