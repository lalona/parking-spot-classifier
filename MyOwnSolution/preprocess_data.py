import argparse
import json
from sklearn.preprocessing import Normalizer
from sklearn import svm
from compare_images_methods2 import getMethods
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
		methods = getMethods()
		features = [m['name'] for m in methods]
		extra_features = ['comparing_with_brig', 'comparing_to_brig']
		features.extend(extra_features)

		print(subset_spaces)

		dataset_train = {'X': [], 'Y': []}
		dataset_test = {'X': [], 'Y': []}
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						for data in spaces_comparisions:
								x = []
								for feature, value in data.items():
										if feature in features:
												x.append(value)
								if space in subset_spaces['train'][parkinglot]:
										dataset_train['X'].append(x)
										if to_categorical_flag:
											dataset_train['Y'].append(to_categorical(int(data['state']), 2))
										else:
											dataset_train['Y'].append(int(data['state']))
								elif space in subset_spaces['test'][parkinglot]:
										dataset_test['X'].append(x)

										if to_categorical_flag:
												dataset_test['Y'].append(to_categorical(int(data['state']), 2))
										else:
												dataset_test['Y'].append(int(data['state']))

		return [dataset_train, dataset_test]

def getNormalizedX(X):
		transformer = Normalizer().fit(X)
		return transformer.transform(X)

def getFeatures():
		methods = getMethods()
		features = [m['name'] for m in methods]
		extra_features = ['comparing_with_brig', 'comparing_to_brig']
		features.extend(extra_features)
		return features

def main():
	parser = argparse.ArgumentParser(description='Select the type of reduced.')
	parser.add_argument("-d", "--dataset-unprocess", type=str, required=True,
												help='Path to the dataset unprocessed.')

	args = vars(parser.parse_args())

	dataset_unprocess_path = args["dataset_unprocess"]

	with open(dataset_unprocess_path) as f:
			dataset_unprocess = json.load(f)

	subset_spaces = getSubsetsSpaces(dataset_unprocess)
	dataset_train, dataset_test = getSubsets(dataset_unprocess, subset_spaces)

	X = dataset_train['X']
	X_test = dataset_test['X']
	y = dataset_train['Y']
	y_test = dataset_test['Y']
	train_ = {'train': {'X': X, 'Y': y}, 'test': {'X': X_test, 'Y': y_test}}

	X = getNormalizedX(X)
	X_test = getNormalizedX(X_test)

	clf = svm.LinearSVC(verbose=True)
	clf.fit(X, y)
	result = clf.score(X_test, y_test)
	print(result)
	filename = 'clf_svm.sav'
	with open(filename, 'wb') as f:
			pickle.dump(clf, f)



	with open('data_training.json', 'w') as outfile:
			json.dump(train_, outfile)




if __name__ == "__main__":
	main()