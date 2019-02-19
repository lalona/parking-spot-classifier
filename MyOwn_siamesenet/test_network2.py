import pickle
import argparse
import os
import json
import ntpath
from preprocess_data import getSubsetsSpaces, getSubsets
from tqdm import tqdm
from keras.models import load_model
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

IMG_HEIGHT = 200
IMG_WIDTH = 200
def getSubsets_info(dataset_unprocess):

		dataset_info = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						for data in spaces_comparisions:
								data_ = {}
								data_['X1'] = (data['comparing_with'])
								data_['X2'] = (data['comparing_to'])
								data_['Y'] = (int(data['state']))
								data_['data'] = (data)
								dataset_info.append(data_)
		return dataset_info

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-m", "--model", type=str, required=True,
												help='Path to the neural net trained.')
		parser.add_argument("-d", "--dataset", type=str, required=True,
												help='Path to dataset.')

		args = vars(parser.parse_args())

		model_path = args["model"]
		dataset_path = args["dataset"]

		model = load_model(model_path)

		with open(dataset_path) as f:
				dataset = json.load(f)


		model_onlypath, model_filename = os.path.split(model_path)
		test_dir = os.path.join(model_onlypath, 'test_info')
		if not os.path.isdir(test_dir):
			os.mkdir(test_dir)

		dataset_info = getSubsets_info(dataset)
		failed_data = []
		count = 0
		total_count = 0

		ave = np.zeros(3)
		std = np.zeros(3) + 1
		X_img = []
		X_img_compare = []

		img_file = ''
		for data_info  in tqdm(dataset_info):
			X_img = []
			X_img_compare = []
			if img_file != data_info['X1']:
				img_file = data_info['X1']
				img = load_img(img_file, target_size=(IMG_WIDTH, IMG_HEIGHT))
				x = img_to_array(img)


			img_file_compare = data_info['X2']
			y = data_info['Y']
			data = data_info['data']

			img_compare = load_img(img_file_compare, target_size=(IMG_WIDTH, IMG_HEIGHT))
			x2 = img_to_array(img_compare)

			X_img.append(x)
			X_img_compare.append(x2)
			X_img = np.array(X_img, dtype="float") / 255.0
			X_img_compare = np.array(X_img_compare, dtype="float") / 255.0
			X = [X_img, X_img_compare]
			#x = np.expand_dims(x, axis=0)
			(ocuppied, empty) = model.predict(X)[0]

			# predicted_y = svm_classifier.predict([x])[0]
			if (y == 1) != (ocuppied > empty):
					failed_data.append(data)
			if count == 1000:
					print('From {} {} failed err % {}'.format(total_count, len(failed_data), len(failed_data) * 100 / total_count))
					count = 0
			count += 1
			total_count += 1

		name_labels = ntpath.basename(dataset_path).split('.')[0]
		failed_info = {'dataset': dataset_path, 'error_por': len(failed_data) * 100 / total_count, 'failed_data': failed_data}
		with open(os.path.join(test_dir, 'error_images_info_{}.json'.format(name_labels)), 'w') as outfile:
				json.dump(failed_info, outfile)




if __name__ == "__main__":
		main()