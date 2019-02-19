import argparse
import json
from sklearn.preprocessing import Normalizer
from sklearn import svm
from tqdm import tqdm
from random import shuffle
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
TRAIN_POR = 80
TEST_POR = 20

def getSubsetsSpaces(dataset_unprocess, train_por=TRAIN_POR, test_por=TEST_POR, parkinglot_limit=100):
	train_subset = {}
	test_subset = {}
	parkinglot_count = 0
	for parkinglot, spaces in dataset_unprocess.items():
			spaces = list(spaces.keys())
			shuffle(spaces)
			train_subset[parkinglot] = []
			test_subset[parkinglot] = []
			if parkinglot_count >= parkinglot_limit:
					test_subset[parkinglot].extend(spaces)
					parkinglot_count += 1
					continue
			len_train_spaces = int(len(spaces) * train_por / 100)
			count = 0
			for s in spaces:
				if count < len_train_spaces:
						train_subset[parkinglot].append(s)
				else:
						test_subset[parkinglot].append(s)
				count += 1
			parkinglot_count += 1
	return {'train': train_subset, 'test': test_subset}

def getSubsets(dataset_unprocess, subset_spaces, to_categorical_flag=False):
		print(subset_spaces)

		dataset_train = []
		dataset_test = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						for data in spaces_comparisions:
								data_ = {}
								data_['X1'] = data['comparing_with']
								data_['X2'] = data['comparing_to']
								if 'comparing_with_state' in data:
										data_['comparing_with_state'] = data['comparing_with_state']
								data_['X2'] = data['comparing_to']
								if to_categorical_flag:
										data_['Y'] = to_categorical(int(data['state']), 2)
								else:
										data_['Y'] = int(data['state'])
								if space in subset_spaces['train'][parkinglot]:
										dataset_train.append(data_)
								elif space in subset_spaces['test'][parkinglot]:
										dataset_test.append(data_)
		return [dataset_train, dataset_test]

def getSubset_dummy(dataset, new_size):
		dataset_dummy = []
		shuffle(dataset)
		positive_count = 0
		positive_empty_count = 0
		positive_occupied_count = 0
		negative_count = 0
		half_size = int(new_size / 2)
		for data in dataset:
				if data['Y'] == 1:
						if positive_count >= half_size:
								continue
						if 'comparing_with_state' in data:
								if data['comparing_with_state'] == '0':
										if positive_empty_count > int(half_size / 2):
											positive_empty_count += 1
								else:
										if positive_occupied_count > int(half_size / 2):
												positive_occupied_count += 1
						positive_count += 1
				else:
						if negative_count >= half_size:
								continue
						negative_count += 1
				dataset_dummy.append(data)
		print('pos: {} neg: {}'.format(positive_count, negative_count))
		return dataset_dummy


def getNormalizedX(X):
		transformer = Normalizer().fit(X)
		return transformer.transform(X)



if __name__ == "__main__":
	main()