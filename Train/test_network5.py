from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

import pickle
import os
from tqdm import tqdm
import json
import ntpath
from keras.layers.core import Layer

def createErrorFile(test_dir, name_labels, failed_images):
		err_file = os.path.join(os.getcwd(), test_dir, 'error_images_info_{}.txt'.format(name_labels))
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
				print('something')
		with open(err_file, 'w') as outfile:
				json.dump(failed_images, outfile)


def errorFileExists(test_dir, name_labels):
		err_file = os.path.join(os.getcwd(), test_dir, 'error_images_info_{}.txt'.format(name_labels))
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
		if os.path.isfile(err_file):
				with open(err_file, 'r') as outfile:
						errors = json.load(outfile)
				if len(errors['image_info']) > 10:
					return True
				else:
						return False
		else:
				return False

def getImagePathPklot(image_info):
		return os.path.join('C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot', image_info["filepath"],
								 image_info["filename"])


def getImagePathCnrpark(image_info):
		return os.path.join("C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/", image_info['filePath'])

def test(images_info_path, model_path, dim, database):

		if database != 'pklot' and database != 'cnrpark':
				print('The database can only be pklot o cnrpark')
				return

		with open(images_info_path, "rb") as fp:  # Unpickling
				images_info_reduced = pickle.load(fp)

		# load the trained convolutional neural network
		print("[INFO] loading network...")

		model = load_model(model_path)
		model.summary()
		model_onlypath, model_filename = os.path.split(model_path)

		failed_images = {'dataset': images_info_path, 'error': 0, 'image_info': []}
		count = 0
		good = 0
		error = 0
		total_count = 0

		name_labels = ntpath.basename(images_info_path).split('.')[0]
		print(name_labels)

		if database == 'pklot':
				getImagePath = getImagePathPklot
		elif database == 'cnrpark':
				getImagePath = getImagePathCnrpark

		for image_info in tqdm(images_info_reduced):
				path_image = getImagePath(image_info)
				# load the image
				image = cv2.imread(path_image)

				# pre-process the image for classification
				image = cv2.resize(image, (dim, dim))
				image = image.astype("float") / 255.0
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)

				# classify the input image
				(empty, ocuppied) = model.predict(image)[0]

				# build the label
				label = "Occupied" if ocuppied > empty else "Empty"
				proba = ocuppied if ocuppied > empty else empty
				label = "{}: {:.2f}%".format(label, proba * 100)

				# esto puede cambiar, de pendiendo con que clase empiece data_paths_train
				# if (int(image_info['state']) == 0) != (ocuppied > empty): # si data_paths_train empieza con 1
				if (int(image_info['state']) == 0) != (ocuppied < empty):  # si data_paths_train empieza con 0
						# draw the label on the image
						image_info['proba_empty'] = str(empty)
						image_info['proba_ocuppied'] = str(ocuppied)
						image_info['whole_path'] = path_image
						failed_images['image_info'].append(image_info)
						error += 1
				else:
						good += 1
				count += 1
				if count == 1000:
						por_error = (len(failed_images['image_info']) * 100) / total_count
						print("In a total of 1000 error: {} total of error images: {} error por.: {}".format(error, len(
								failed_images['image_info']), por_error))
						count = 0
						error = 0
						good = 0
				total_count += 1

		por_error = (len(failed_images['image_info']) * 100) / len(images_info_reduced)

		print("In a total of {} error: {} error por.: {}".format(len(images_info_reduced), len(
				failed_images['image_info']), por_error))

		failed_images['error'] = por_error
		name_labels = ntpath.basename(images_info_path).split('.')[0]
		test_dir = os.path.join(model_onlypath, 'test_info')
		createErrorFile(test_dir, name_labels, failed_images)

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-e", "--experiment-dir", required=True,
										help="The path to the experiment dir.")

		images_info_path_pklot = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\pklot_labels_reduced_comparing-images_70.txt'
		images_info_path_cnrpark = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\cnrpark_labels_reduced_comparing-images_70.txt'

		args = vars(ap.parse_args())

		experiment_dir = args['experiment_dir']

		experiment_specs_training_params_path = os.path.join(experiment_dir, 'training_params.json')

		if os.path.isfile(experiment_specs_training_params_path):
				with open(os.path.join(experiment_specs_training_params_path), 'r') as outfile:
						training_params = json.load(outfile)
		else:
				print('You need to run the training first.')
				return

		dim = training_params['dim']

		experiments = [ name for name in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, name)) ]

		if len(experiments) == 0:
				print('You need to run the training first.')
				return

		for experiment in experiments:
			model_path = os.path.join(experiment_dir, experiment, 'parking_classification.model')
			model_onlypath, model_filename = os.path.split(model_path)
			name_labels = ntpath.basename(images_info_path_pklot).split('.')[0]
			test_dir = os.path.join(model_onlypath, 'test_info')
			createErrorFile(test_dir, name_labels, failed_images)

			print(experiment)
			test




if __name__ == "__main__":
		main()

