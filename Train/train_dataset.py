import cv2
from keras.preprocessing.image import img_to_array
import argparse
import pickle
import os
from tqdm import tqdm
import random
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from malexnet import mAlexNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import json
from googlenet import create_googlenet


def generator(data, batch_size):
	while True:
		images = []
		labels = []

		for i in range(batch_size):
			# choose random index in features
			d = random.choice(data)
			#with open(data[index]['path'], "rb") as fp:  # Unpickling
			#	image = pickle.load(fp)
			image = np.load(d['path'])
			images.append(image)
			labels.append(d['y'])
		labels = np.array(labels)
		labels = to_categorical(labels, num_classes=2)
		#print(labels)
		yield np.array(images), labels

labels_file = "C:\\Eduardo\\Level1\\DeepLearning_Keras\\my_code\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\cnrpark_labels_reduced_comparing-images_60.txt"

#path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

EPOCHS = 20
INIT_LR = 1e-3
BS = 200

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-d", "--dataset", required=True,
										help="path to input dataset")
		ap.add_argument("-m", "--model", required=True,
										help="path to output model")
		ap.add_argument("-p", "--plot", type=str, default="plot.png",
										help="path to output loss/accuracy plot")
		args = vars(ap.parse_args())

		dataset_path = args['dataset']
		model_path = args['model']

		# Helps to get the average of height and width in the images
		"""
		if not os.path.isfile(labels_file):
				print('Insert a valid file')
				return

		with open(labels_file, "rb") as fp:  # Unpickling
				images_info_reduced = pickle.load(fp)		
		
		height = 0
		width = 0
		# loop over the input images
		for image_info in tqdm(images_info_reduced):
				# load the image, pre-process it, and store it in the data list
				image = cv2.imread(os.path.join(path_patches, image_info['filePath']))
				h, w, c = image.shape
				height += h
				width += w
		average_height = height / (len(images_info_reduced))
		average_width = width / (len(images_info_reduced))

		print('Cnrpark average height: {} average width: {}'.format(average_height, average_width))
		"""

		# initialize the model
		print("[INFO] compiling model...")
		#model = LeNet.build(width=70, height=70, depth=3, classes=2)
		model, name = mAlexNet.build(width=70, height=70, depth=3, classes=2)
		#model, name = create_googlenet(width=70, height=70, depth=3, classes=2)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
									metrics=["accuracy"])

		# train the network
		print("[INFO] training network...")
		#with open(dataset_path, "rb") as fp:  # Unpickling
		#		data = pickle.load(fp)
		with open(dataset_path, "r") as infile:
				data = json.load(infile)

		print(model.summary())

		"""
		H = model.fit_generator(generator=generator(data=data['train'], batch_size=BS),
														validation_data=generator(data=data['test'], batch_size=BS), steps_per_epoch=len(data['train']) // BS,
														validation_steps=len(data['test']) // BS,
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
		plt.savefig(args["plot"])
		"""

if __name__ == "__main__":
		main()

