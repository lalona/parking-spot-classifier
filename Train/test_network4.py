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

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--model", required=True,
										help="path to trained model model")
		ap.add_argument("-dim", "--dimension", type=int, required=True,
										help='The size of width and height.')
		ap.add_argument("-d", "--dataset", type=str, required=True,
										help='It can be pklot or cnrpark.')

		args = vars(ap.parse_args())

		dataset = args['dataset']

		dim = args['dimension']

		if dataset != 'pklot' and dataset != 'cnrpark':
				print('The dataset can only be pklot o cnrpark')
				return

		if dataset == 'pklot':
				dataset_directory = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets\\test_only\\pklot_labels_reduced_comparing-images_50'
		else:
				dataset_directory = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets\\test_only\\cnrpark_labels_reduced_comparing-images_50'

		# load the trained convolutional neural network
		print("[INFO] loading network...")
		model_path = args["model"]
		model = load_model(model_path,custom_objects={'PoolHelper': PoolHelper})

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
						failed_images.append(row['path'])
						error_count += 1
				count += 1
				if count >= 1000:
						print('% err: {}'.format(error_count * 100 / total_count))
						count = 0
				total_count += 1
		test_stadistics['failed_images'] = failed_images
		test_stadistics['error_por'] = (len(failed_images) * 100) / len(predictions)

		dataset_tested = os.path.basename(os.path.normpath(dataset_directory))
		model_onlypath, model_filename = os.path.split(model_path)
		test_dir = os.path.join(model_onlypath, 'test_info')
		createErrorFile(test_dir, dataset_tested, test_stadistics)

class PoolHelper(Layer):

		def __init__(self, **kwargs):
				super(PoolHelper, self).__init__(**kwargs)

		def call(self, x, mask=None):
				return x[:, 1:, 1:]

		def get_config(self):
				config = {}
				base_config = super(PoolHelper, self).get_config()
				return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
		main()

