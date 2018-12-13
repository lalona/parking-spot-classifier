"""
This should receive the a file with the paths of the images for training his state an images for the validation
"""
import random
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import json
from googlenet import create_googlenet
from malexnet import mAlexNet
from datetime import datetime

IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_CHANNEL = 3
NUM_CLASSES = 2

labels_file = "C:\\Eduardo\\Level1\\DeepLearning_Keras\\my_code\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\cnrpark_labels_reduced_comparing-images_60.txt"

# path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

EPOCHS = 20
INIT_LR = 1e-3
BATCH_SIZE = 100

def create_aug_gen(in_gen, image_gen):

	for in_x, in_y in in_gen:
		g_x = image_gen.flow(255*in_x, batch_size=in_x.shape[0])
		yield next(g_x)/255.0, in_y


# fuction only to read image from file
def get_image(data):
		# Read image and appropiate traffic light color
		image = cv2.imread(os.path.join(data['path']))
		image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
		state = data['y']
		return [image, state]

# generator function to return images batchwise
def generator(data):
	while True:
		# Randomize the indices to make an array
		indices_arr = np.random.permutation(len(data))
		for batch in range(0, len(indices_arr), BATCH_SIZE):
			# slice out the current batch according to batch-size
			current_batch = indices_arr[batch:(batch + BATCH_SIZE)]
			# initializing the arrays, x_train and y_train
			x_train = np.empty([0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
			y_train = np.empty([0], dtype=np.int32)
			for i in current_batch:
				# get an image and its corresponding color for an traffic light
				[image, state] = get_image(data[i])
				# Appending them to existing batch
				x_train = np.append(x_train, [image], axis=0)
				y_train = np.append(y_train, [state])
			y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
			yield (x_train, y_train)

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-d", "--dataset", required=True,
										help="path to input dataset")

		args = vars(ap.parse_args())

		dataset_path = args['dataset']
		train_results_direcotry = os.path.dirname(os.path.abspath(dataset_path))
		train_results_direcotry = os.path.join(train_results_direcotry, datetime.now().strftime("%d-%m-%Y_H:M"))
		print(train_results_direcotry)

		model_path = os.path.join(train_results_direcotry, 'parking_classification.model')
		plot_path = os.path.join(train_results_direcotry, 'plot.png')
		train_details_path = os.path.join(train_results_direcotry, 'details.json')
		with open(dataset_path, "r") as infile:
				data = json.load(infile)


		# initialize the model
		print("[INFO] compiling model...")
		# model = LeNet.build(width=70, height=70, depth=3, classes=2)
		model, architecture_name = mAlexNet.build(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, depth=IMAGE_CHANNEL, classes=NUM_CLASSES)
		# model, arquitecture_name = create_googlenet(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, depth=IMAGE_CHANNEL, classes=NUM_CLASSES)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
									metrics=["accuracy"])

		image_gen = ImageDataGenerator(rotation_range=15,
																	 width_shift_range=0.1,
																	 height_shift_range=0.1,
																	 shear_range=0.01,
																	 zoom_range=[0.9, 1.25],
																	 horizontal_flip=True,
																	 vertical_flip=False,
																	 fill_mode='reflect',
																	 data_format='channels_last',
																	 brightness_range=[0.5, 1.5])
		train_gen = generator(data['train'])
		cur_train_gen = create_aug_gen(train_gen, image_gen)
		test_gen = generator(data['test'])
		cur_test_gen = create_aug_gen(test_gen, image_gen)


		# train the network
		print("[INFO] training network...")
		# with open(dataset_path, "rb") as fp:  # Unpickling
		#		data = pickle.load(fp)
		with open(dataset_path, "r") as infile:
				data = json.load(infile)

		print(model.summary())

		H = model.fit_generator(generator=cur_train_gen,
														validation_data=cur_test_gen,
														steps_per_epoch=len(data['train']) // BATCH_SIZE,
														validation_steps=len(data['test']) // BATCH_SIZE,
														epochs=EPOCHS, verbose=1)
		# save the model to disk
		print("[INFO] serializing network...")
		model.save(model_path)

		# plot the training loss and accuracy
		plt.style.use("ggplot")
		plt.figure()
		N = EPOCHS
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
				'arquitecture': architecture_name
		}
		with open(os.path.join(train_details_path), 'w') as outfile:
				json.dump(details, outfile)


if __name__ == "__main__":
		main()

