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
import cv2
from keras.models import Model
from scipy.spatial.distance import cdist
import msvcrt as m
IMG_HEIGHT = 200
IMG_WIDTH = 200

#python test_network101.py -m C:\Eduardo\ProyectoFinal\Proyecto\ProyectoFinal\Train\test_folder\02-28\cnrpark_labels_reduced_comparing-images_70-15576v-13918nv_complementary-mio-td-1924v-3582nv_mgv8\parking_classification.model -d C:\Eduardo\ProyectoFinal\Proyecto\ProyectoFinal\MyOwn_siamesenet\dataset_comp_pklot_labels_reduced_comparing-images_70.json


def createErrorFile(test_dir, name_labels, failed_images, distance_param, layer_name):
		err_file = os.path.join(os.getcwd(), test_dir, 'error_images_info_101_{}_{}_{}.txt'.format(distance_param, layer_name, name_labels))
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
				print('something')
		with open(err_file, 'w') as outfile:
				json.dump(failed_images, outfile)


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


def getSubsets_info2(dataset_unprocess):
		dataset_info = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						x1_e = spaces_comparisions['comparing_with_empty']
						x1_o = spaces_comparisions['comparing_with_occupied']
						for data in spaces_comparisions['comparissions']:
								data_ = {}
								data_['X1_e'] = (x1_e)
								data_['X1_o'] = (x1_o)
								data_['X2'] = (data['comparing_to'])
								data_['Y'] = (int(data['state']))
								data_['data'] = (data)
								dataset_info.append(data_)
		return dataset_info

def getPred(path_image, dim, intermediate_layer_model):
		image = cv2.imread(path_image)
		orig = image.copy()

		# pre-process the image for classification
		image = cv2.resize(image, (dim, dim))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		intermediate_output = intermediate_layer_model.predict(image)
		return intermediate_output


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

		dataset_info = getSubsets_info2(dataset)
		failed_data = []
		total_count = 0
		count = 0

		img_file_e = ''
		img_file_o = ''
		model.summary()
		layer_name = 'dense_1'
		intermediate_layer_model = Model(inputs=model.input,
																		 outputs=model.get_layer(layer_name).output)

		distance_param ='cityblock'
		test_info = {'model_path': model_path, 'dataset_path': dataset_path, 'layer_name': layer_name, 'error': 0, 'failed_data': []}

		for data_info  in tqdm(reversed(dataset_info)):
			if img_file_e != data_info['X1_e']:
				img_file_e = data_info['X1_e']
				img_file_o = data_info['X1_o']
				pred1 = getPred(img_file_e, IMG_WIDTH, intermediate_layer_model)
				pred2 = getPred(img_file_o, IMG_WIDTH, intermediate_layer_model)


			img_file_compare = data_info['X2']
			y = data_info['Y']
			data = data_info['data']
			pred3 = getPred(img_file_compare, IMG_WIDTH, intermediate_layer_model)
			dist_e = cdist(pred1, pred3, distance_param)
			dist_o = cdist(pred2, pred3, distance_param)
			# predicted_y = svm_classifier.predict([x])[0]
			#print("dist: {} state: {}".format(dist, y))

			if dist_e >= dist_o and y == 0:
				failed_data.append(data)
			if dist_o >= dist_e and y == 1:
				failed_data.append(data)

			"""
			if (y == 1) != (ocuppied > empty):
					failed_data.append(data)
			"""
			if count == 1000:
					print('From {} {} failed err % {}'.format(total_count, len(failed_data), len(failed_data) * 100 / total_count))
					count = 0
					#break
			count += 1
			total_count += 1

		test_info['error'] = len(failed_data) * 100 / total_count
		test_info['failed_data'] = failed_data

		name_labels = ntpath.basename(dataset_path).split('.')[0]
		failed_info = {'dataset': dataset_path, 'error_por': len(failed_data) * 100 / total_count, 'failed_data': failed_data}
		test_dir = os.path.join(model_onlypath, 'test_info')
		createErrorFile(test_dir, name_labels, failed_info, distance_param, layer_name)




if __name__ == "__main__":
		main()