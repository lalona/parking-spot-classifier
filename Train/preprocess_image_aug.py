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
#from orig_googlenet import create_googlenet
from googlenet import create_googlenet
from malexnet import mAlexNet
from datetime import datetime
import pandas as pd
IMAGE_HEIGHT = 140
IMAGE_WIDTH = 140
IMAGE_CHANNEL = 3
NUM_CLASSES = 2

labels_file = "C:\\Eduardo\\Level1\\DeepLearning_Keras\\my_code\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\cnrpark_labels_reduced_comparing-images_60.txt"

# path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

EPOCHS = 20
INIT_LR = 1e-3
BATCH_SIZE = 200

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-d", "--dataset", required=True,
										help="path to input dataset")

		args = vars(ap.parse_args())

		dataset_path = args['dataset']
		#train_results_direcotry = os.path.dirname(os.path.abspath(dataset_path))
		train_results_direcotry = os.path.join(dataset_path, datetime.now().strftime("%d-%m-%Y_%H-%M"))
		print(train_results_direcotry)
		model_path = os.path.join(train_results_direcotry, 'parking_classification.model')
		plot_path = os.path.join(train_results_direcotry, 'plot.png')
		train_details_path = os.path.join(train_results_direcotry, 'details.json')

		# initialize the model
		print("[INFO] compiling model...")
		# model = LeNet.build(width=70, height=70, depth=3, classes=2)
		#model, architecture_name = mAlexNet.build(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, depth=IMAGE_CHANNEL, classes=NUM_CLASSES)
		model, architecture_name = create_googlenet(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, depth=IMAGE_CHANNEL, classes=NUM_CLASSES)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
									metrics=["accuracy"])

		image_gen = ImageDataGenerator(rescale=1./255, rotation_range=90, horizontal_flip=True, vertical_flip=True)
		traindf = pd.read_csv(os.path.join(dataset_path, 'data_paths_train.csv'))
		traindf_aug = pd.read_csv(os.path.join(dataset_path, 'data_paths_aug.csv'))
		traindf_extended = pd.concat([traindf, traindf_aug])
		train_generator = image_gen.flow_from_dataframe(dataframe=traindf, directory=None, x_col='path', y_col='y', shuffle= True,
																									has_ext=True, class_mode="categorical", target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
																									batch_size=BATCH_SIZE)

		for y in traindf.columns:
				print("Column: {} Type: {}".format(y, traindf[y].dtype))

		image_gen2 = ImageDataGenerator(rescale=1. / 255)
		testdf = pd.read_csv(os.path.join(dataset_path, 'data_paths_test.csv'))
		print("The size of training is: {} and test is: {}".format(traindf.shape, testdf.shape))
		test_generator = image_gen2.flow_from_dataframe(dataframe=testdf, directory=None, x_col='path', y_col='y', shuffle= True,
																										has_ext=True, class_mode="categorical", target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
																										batch_size=BATCH_SIZE)

		# train the network
		print("[INFO] training network...")
		# with open(dataset_path, "rb") as fp:  # Unpickling


		#print(model.summary())

		STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
		STEP_SIZE_VALID = test_generator.n // test_generator.batch_size
		H = model.fit_generator(generator=train_generator,
														validation_data=test_generator,
														steps_per_epoch=STEP_SIZE_TRAIN,
														validation_steps=STEP_SIZE_VALID,
														epochs=EPOCHS, verbose=1)
		os.mkdir(train_results_direcotry)
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

