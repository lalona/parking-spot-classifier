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
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def createErrorFile(test_dir, dataset_tested, failed_images):
		err_file = os.path.join(os.getcwd(), test_dir, 'real_error_info_{}.txt'.format(dataset_tested))
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
				print('something')

		with open(err_file, 'w') as outfile:
				json.dump(failed_images, outfile)


def errorFileExists(test_dir, dataset_tested):
		err_file = os.path.join(os.getcwd(), test_dir, 'real_error_info_{}.txt'.format(dataset_tested))
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
		if os.path.isfile(err_file):
				with open(err_file, 'r') as outfile:
						errors = json.load(outfile)
				if len(errors['failed_images']) > 10:
						return (True, err_file)
				else:
						return (False, err_file)
		else:
				return (False, err_file)


def test(model_path, dataset_directory, dim):

		dataset_tested = os.path.basename(os.path.normpath(dataset_directory))
		model_onlypath, model_filename = os.path.split(model_path)
		test_dir = os.path.join(model_onlypath, 'test_info')
		err_file_exists, err_file = errorFileExists(test_dir, dataset_tested)

		if err_file_exists:
				print('The test was already made {}'.format(err_file))
				return

		# load the trained convolutional neural network
		print("[INFO] loading network...")
		model = load_model(model_path)

		df_test = pd.read_csv(os.path.join(dataset_directory, 'data_paths_test.csv'))
		test_datagen = ImageDataGenerator(rescale=1. / 255.)
		datagen = ImageDataGenerator(rescale=1. / 255.)

		train_generator = datagen.flow_from_dataframe(dataframe=df_test, x_col="path", y_col="y", directory=None,
																										class_mode="categorical", target_size=(dim, dim),
																										batch_size=1)

		test_generator = test_datagen.flow_from_dataframe(dataframe=df_test, x_col="path", y_col=None, directory=None,
																											target_size=(dim, dim), batch_size=1, shuffle=False, class_mode=None)

		STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
		test_generator.reset()
		pred = model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
		predicted_class_indices = np.argmax(pred, axis=1)
		labels = (train_generator.class_indices)
		print(labels)
		labels = dict((v, k) for k, v in labels.items())

		predictions = [labels[k] for k in predicted_class_indices]
		filenames = test_generator.filenames

		total_count = 0
		count = 0
		error_count = 0
		test_stadistics = {'dataset_tested': dataset_directory, 'error_por': 0, 'failed_images': []}
		failed_images = []
		filenames_predictions = {}
		for filename, prediction in zip(filenames, predictions):
				filenames_predictions[filename] = prediction
		for (index, row) in df_test.iterrows():
				pred = filenames_predictions[row['path']]
				if row['y'] != pred:
						failed_images.append({'file': row['path'], 'state': row['y']})
						error_count += 1
				count += 1
				if count >= 1000:
						print('% err: {}'.format(error_count * 100 / total_count))
						count = 0
				total_count += 1
		test_stadistics['failed_images'] = failed_images
		test_stadistics['error_por'] = (len(failed_images) * 100) / len(predictions)


		createErrorFile(test_dir, dataset_tested, test_stadistics)


def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-e", "--experiment-dir", required=True,
										help="The path to the experiment dir.")

		dataset_directory_pklot = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets\\test_only\\pklot_labels_reduced_comparing-images_70'
		dataset_directory_cnrpark = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets\\test_only\\cnrpark_labels_reduced_comparing-images_70'

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

		experiments = [name for name in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, name))]

		if len(experiments) == 0:
				print('You need to run the training first.')
				return

		for experiment in experiments:
				print('Testing {}'.format(experiment))
				model_path = os.path.join(experiment_dir, experiment, 'parking_classification.model')
				test(model_path, dataset_directory_pklot, dim)

		for experiment in experiments:
				print('Testing {}'.format(experiment))
				model_path = os.path.join(experiment_dir, experiment, 'parking_classification.model')
				test(model_path, dataset_directory_cnrpark, dim)

if __name__ == "__main__":
		main()

