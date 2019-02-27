from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import os
import random
import argparse
from keras.preprocessing.image import load_img, img_to_array

IMG_DIM = 200

def getImage(img_path):
		img = load_img(img_path, target_size=(IMG_DIM, IMG_DIM))
		x = img_to_array(img)
		return x

class Dataset(object):
	#images_train = np.array([])
	#images_test = np.array([])
	#labels_train = np.array([])
	#labels_test = np.array([])
	unique_train_label = []
	map_train_label = dict()
	map_test_label = dict()

	def _get_siamese_similar_pair(self):
		label = random.choice(self.unique_train_label)
		l, r = random.sample(self.map_train_label[label], 2)
		return getImage(l), getImage(r), 1

	def _get_siamese_dissimilar_pair(self):
		#print(len(self.unique_train_label))
		while True:
			label_l, label_r = random.sample(self.unique_train_label, 2)
			if label_l != label_r:
					break
		#label_l, label_r = self.unique_train_label
		l = random.choice(self.map_train_label[label_l])
		r = random.choice(self.map_train_label[label_r])
		return getImage(l), getImage(r), 0

	def _get_siamese_pair(self):
		if np.random.random() < 0.5:
			return self._get_siamese_similar_pair()
		else:
			return self._get_siamese_dissimilar_pair()

	def get_siamese_batch(self, n):
		idxs_left, idxs_right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_siamese_pair()
			idxs_left.append(l)
			idxs_right.append(r)
			labels.append(x)
		left = np.array(idxs_left) / 255.0
		right = np.array(idxs_right) / 255.0
		print(left.shape)
		return left, right, np.expand_dims(labels, axis=1)



def getMapLabel(df):
		map_label = dict()
		for index, row in df.iterrows():
				y = row['y']
				img_path = row['path']
				if y not in map_label:
						map_label[y] = []
				map_label[y].append(img_path)
		return map_label


class ParkinglotDataset(Dataset):
	def __init__(self, dataset_directory):
		print("===Loading MNIST Dataset===")
		df_train = pd.read_csv(os.path.join(dataset_directory, 'data_paths_train.csv'))
		df_test = pd.read_csv(os.path.join(dataset_directory, 'data_paths_test.csv'))
		map_train = getMapLabel(df_train)
		map_test = getMapLabel(df_test)

		#(self.images_train, self.labels_train), (self.images_test, self.labels_test) = mnist.load_data()
		#self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
		#self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		#self.labels_train = np.expand_dims(self.labels_train, axis=1)
		#self.unique_train_label = np.unique(self.labels_train)
		#self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		self.map_train_label = map_train
		self.map_test_label = map_test
		self.unique_train_label = list(map_train.keys())

		
if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
									help="path to input dataset directory")

	args = vars(ap.parse_args())

	dataset_directory = args['dataset']

	a = ParkinglotDataset(dataset_directory)
	batch_size = 4
	ls, rs, xs = a.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2)
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("max:", np.squeeze(l, axis=2).max())
		axarr[idx, 0].imshow(np.squeeze(l, axis=2))
		axarr[idx, 1].imshow(np.squeeze(r, axis=2))
	plt.show()