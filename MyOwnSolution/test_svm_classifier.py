import pickle
import argparse
import os
import json
import ntpath
from preprocess_data import getSubsetsSpaces, getSubsets,getNormalizedX
from tqdm import tqdm
from compare_images_methods2 import getMethods

def getSubsets_info(dataset_unprocess):
		methods = getMethods()
		features = [m['name'] for m in methods]
		extra_features = ['comparing_with_brig', 'comparing_to_brig']
		features.extend(extra_features)

		dataset_info = {'X': [], 'Y': [], 'data': []}
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						for data in spaces_comparisions:
								x = []
								for feature, value in data.items():
										if feature in features:
												x.append(value)
												dataset_info['X'].append(x)
												dataset_info['Y'].append(int(data['state']))
												dataset_info['data'].append(data)


		return dataset_info

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-c", "--classifier", type=str, required=True,
												help='Path to the svm classifier.')
		parser.add_argument("-d", "--dataset", type=str, required=True,
												help='Path to dataset.')

		args = vars(parser.parse_args())

		svm_classifier_path = args["classifier"]
		dataset_path = args["dataset"]

		with open(svm_classifier_path, "rb") as fp:  # Unpickling
				svm_classifier = pickle.load(fp)

		with open(dataset_path) as f:
				dataset = json.load(f)


		svm_classifier_onlypath, svm_classifier_filename = os.path.split(svm_classifier_path)
		test_dir = os.path.join(svm_classifier_onlypath, 'test_info')

		dataset_info = getSubsets_info(dataset)
		failed_data = []
		count = 0
		total_count = 0
		for x, y, data in zip(dataset_info['X'], dataset_info['Y'], dataset_info['data']):
			predicted_y = svm_classifier.predict([x])[0]
			if predicted_y != y:
					failed_data.append(data)
			if count == 10000:
					print('From {} {} failed err % {}'.format(total_count, len(failed_data), len(failed_data) * 100 / total_count))
					count = 0
			count += 1
			total_count += 1

		name_labels = ntpath.basename(dataset_path).split('.')[0]
		failed_info = {'dataset': dataset_path, 'error_por': len(failed_data) * 100 / total_count, 'failed_data': failed_data}
		with open(os.path.join(test_dir, 'error_images_info_{}.txt'.format(name_labels)), 'w') as outfile:
				json.dump(failed_info, outfile)




if __name__ == "__main__":
		main()