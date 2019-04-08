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


def getImage(img_path, img_dim):
		img = load_img(img_path, target_size=(img_dim, img_dim))
		x = img_to_array(img)
		return x

class Dataset(object):
	unique_train_label = []
	map_train_label = dict()
	map_test_label = dict()

	def _get_siamese_similar_pair(self):
		label = random.choice(self.unique_train_label)
		l, r = random.sample(self.map_train_label[label], 2)
		return getImage(l, self.img_dim), getImage(r, self.img_dim), 1

	def _get_siamese_dissimilar_pair(self):
		while True:
			label_l, label_r = random.sample(self.unique_train_label, 2)
			if label_l != label_r:
					break
		l = random.choice(self.map_train_label[label_l])
		r = random.choice(self.map_train_label[label_r])
		return getImage(l, self.img_dim), getImage(r, self.img_dim), 0

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
	def __init__(self, dataset_directory, img_dim):
		print("===Loading MNIST Dataset===")
		df_train = pd.read_csv(os.path.join(dataset_directory, 'data_paths_train.csv'))
		df_test = pd.read_csv(os.path.join(dataset_directory, 'data_paths_test.csv'))
		self.size_training = df_train['path'].sum()

		map_train = getMapLabel(df_train)
		map_test = getMapLabel(df_test)

		self.map_train_label = map_train
		self.map_test_label = map_test
		self.unique_train_label = list(map_train.keys())
		self.img_dim = img_dim

		
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