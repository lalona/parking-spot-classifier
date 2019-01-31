

"""
Por cada fecha dentro del conjunto de imagenes se van a escoger ciertos cajones
para entrenamiento, validación y prueba. Se hace en cada fecha porque
las imagenes de un espacio tomada por una camara en cierto día, en sus diferentes
horarios no pueden aparecer en más de uno de los tres subconjuntos (entrenamiento, validacion y prueba)
esto se debe a que las imagenes para un espacio en los diferentes horarios pueden contener
el mismo automovil y ser muy similares, lo que provocaria que se estaria probando o validando con imagenes
muy similares a las usadas en el entrenamiento.

Una vez que se sabe que imagenes corresponden a cada subcconjunto, en un archivo json se guarda el path y el estado de las
imagenes en su respectivo subconjunto
	El archivo json contiene a {train:[], test[]} donde en cada lista se guarda por cada imagen un directorio
	de la siguiente forma {path: 'image_path', state: '0 o 1'}

Tambien se guarda un resumen de la informacion, que tantos tienes son vacios y ocupados, que tantos son en dias
soleados, nublados o lluviosos
"""
import csv
import pickle
import argparse
import os
import math
from random import shuffle
from itertools import groupby
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
import ntpath
from keras.utils import to_categorical
import numpy as np
import json
import random
from save_subdatasets_utils import save_subset
import msvcrt as m
TRAIN_SIZE = 70
TEST_SIZE = 30

def preprocessImage(image_path, image_dim):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(image_path)
	image = cv2.resize(image, (image_dim, image_dim))
	return img_to_array(image)


def extractUniqueItemsByKey(list, key):
		"""
		This will take a list and sorted by the key and
		then it will return the list with just the elements from that
		key without duplicates
		:param list:
		:param key:
		:return:
		"""
		list.sort(key=lambda x: x[key])
		return [k for k, v in groupby(list, key=lambda x: x[key])]

def getSizeByPorcentage(size, porcentage):
		return size * (porcentage / 100)

def asignNumberSpacesForSubsets(len_spaces):
		"""
		This asign the number of spaces that each subset is going to take
		:param len_spaces: the total spaces in the whole dataset
		:return: a list containing in position 0,1 and 2 the training, validation and test number of spaces respectively
		"""
		train_size = getSizeByPorcentage(len_spaces, TRAIN_SIZE)
		test_size = getSizeByPorcentage(len_spaces, TEST_SIZE)

		sizes = [round(train_size), round(test_size)]

		i = -1
		while sum(sizes) > len_spaces:
			sizes[i] = sizes[i] - 1
			if i < ((len(sizes)) * -1):
					i = -1
			i -= 1
		return sizes

def getSpacesSubsets(spaces, spaces_distribution):
		"""
		This will return a dictionary where the key will be the number of space
		and the value is the type of subset asigned to that space
		:param spaces:
		:param spacesDistribution:
		:return:
		"""
		shuffle(spaces)
		subsets = {}
		i = 0
		for s in spaces:
				if i < spaces_distribution[0]:
						type = 'train'
				elif i >= spaces_distribution[0]:
						type = 'test'
				subsets[s] = type
				i += 1
		return subsets

path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-f", "--filename", type=str, required=True, help='Path to the file the contains the dictionary with the info of the dataset reduced.')

		args = vars(parser.parse_args())

		info_filename = args["filename"]

		# the dataset filename is gonna be called the same instead of labels it will have dataset
		dataset_filename = info_filename.replace("labels", "dataset")
		dataset_directory = ntpath.basename(dataset_filename).split('.')[0] + "-balance" + "_csv"
		subsets_path = "C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets"
		subsets_path = os.path.join(subsets_path, dataset_directory)

		if os.path.isdir(subsets_path):
				print('This subset was already made')
				#return
		else:
				print("Make dir: {}".format(subsets_path))
				os.mkdir(subsets_path)

		if not os.path.isfile(info_filename):
				print('Insert a valid file')
				return

		with open(info_filename, "rb") as fp:   # Unpickling
				images_info_reduced = pickle.load(fp)

		spaces = extractUniqueItemsByKey(images_info_reduced, 'space')

		spaces_distribution = asignNumberSpacesForSubsets(len(spaces))

		# For each day i would select randoms spaces for each subset (training, validation, test)
		images_info_reduced.sort(key=lambda x: x['date'])
		subsets = {'train': [], 'test': []}
		current_date = images_info_reduced[0]['date']
		spaces_subsets = getSpacesSubsets(spaces, spaces_distribution)

		data_paths = {'train': [], 'test': []}


		for image_info in tqdm(images_info_reduced):
				if current_date != image_info['date']:
						current_date = image_info['date']
						spaces_subsets = getSpacesSubsets(spaces, spaces_distribution)
				# this decides if the spaces is going to be train or test
				subset = spaces_subsets[image_info['space']]
				image_path = os.path.join(path_patches, image_info['filePath'])
				data_path_state = {'path': image_path, 'y': image_info['state']}
				data_paths[subset].append(data_path_state)
				subsets[subset].append(image_info)

		empty_count = 0
		ocuppied_count = 0
		for train_image in data_paths['train']:
				if train_image['y'] == '0':
					empty_count += 1
				else:
					ocuppied_count += 1
		print("Train images empty state: {} occupied state: {}".format(empty_count, ocuppied_count))
		# si las imagenes en entrenamiento están desbalanceada, esto quiere decir que
		# tienen más elementos vacios o ocupados, se balancearan de la siguiente forma
		# en caso de que las imagenes con el espacio vacio sean mayores se pasara el exceso
		# a las imagenes de prueba
		# en caso de que las imagenes de espacios ocupados sea mayor, las imagenes de espacios
		# vacios se aumentara con imagenes del conjunto de entrenamiento
		if empty_count > ocuppied_count:
				train_paths = data_paths['train']
				i = 0
				while i < (empty_count - ocuppied_count):
						random_index = random.randint(0, len(train_paths))
						if train_paths[random_index]['y'] == '0':
								data_path_state = train_paths.pop(random_index)
								data_paths['test'].append(data_path_state)
								image_info = subsets['train'].pop(random_index)
								subsets['test'].append(image_info)
								i += 1
				data_paths['train'] = train_paths
		elif ocuppied_count > empty_count:
				test_paths = data_paths['test']
				i = 0
				while i < (ocuppied_count - empty_count):
						random_index = random.randint(0, len(test_paths))
						if test_paths[random_index]['y'] == '0':
								data_path_state = test_paths.pop(random_index)
								data_paths['train'].append(data_path_state)
								image_info = subsets['test'].pop(random_index)
								subsets['train'].append(image_info)
								i += 1
				data_paths['test'] = test_paths


		for key, value in subsets.items():
				spacesIn = extractUniqueItemsByKey(value, 'space')
				empty_count = 0
				ocuppied_count = 0
				overcast_count = 0
				sunny_count = 0
				rainy_count = 0
				for v in value:
						if v['state'] == '0':
								empty_count += 1
						else:
								ocuppied_count += 1
						if v['weather'] == 'OVERCAST':
								overcast_count += 1
						elif v['weather'] == 'SUNNY':
								sunny_count += 1
						elif v['weather'] == 'RAINY':
								rainy_count += 1
				info = "Subset {} size: {} from {} spaces - empty: {} ocuppied: {} - cloudy: {} sunny: {} rainy: {}".format(
						key,
						len(
								value),
						len(
								spacesIn),
						empty_count,
						ocuppied_count,
						overcast_count,
						sunny_count,
						rainy_count)
				print(info)
				with open(os.path.join(subsets_path, 'data_info.txt'), "a") as finfo:
						finfo.write(info)

		#with open(os.path.join(subsets_path, 'data_paths.txt'), "wb") as fp:  # Pickling
		#		pickle.dump(data_paths, fp)
		with open(os.path.join(subsets_path, 'data_paths.txt'), 'w') as outfile:
				json.dump(data_paths, outfile)

		save_subset(data_paths['train'], dataset_directory, 'train')
		save_subset(data_paths['test'], dataset_directory, 'test')
		"""
		toCSV = data_paths['train']
		keys = toCSV[0].keys()
		with open(os.path.join(subsets_path, 'data_paths_train.csv'), 'w') as output_file:
				#dict_writer = csv.DictWriter(output_file, delimiter=',', lineterminator='\n', fieldnames=keys)
				dict_writer = csv.DictWriter(output_file, fieldnames=keys)
				dict_writer.writeheader()
				dict_writer.writerows(toCSV)

		toCSV2 = data_paths['test']
		keys2 = toCSV2[0].keys()
		with open(os.path.join(subsets_path, 'data_paths_test.csv'), 'w') as output_file:
				dict_writer = csv.DictWriter(output_file, delimiter=',', lineterminator='\n', fieldnames=keys2)
				dict_writer = csv.DictWriter(output_file, fieldnames=keys)
				dict_writer.writeheader()
				dict_writer.writerows(toCSV2)
		"""

if __name__ == "__main__":
		main()












