from itertools import groupby
import os
import csv
from random import shuffle
from tqdm import tqdm
import pickle

base_subsets_path = "C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets"


def getSizeByPorcentage(size, porcentage):
		return size * (porcentage / 100)


def asignNumberSpacesForSubsets(len_spaces, train_size, test_size):
		"""
		This asign the number of spaces that each subset is going to take
		:param len_spaces: the total spaces in the whole dataset
		:return: a list containing in position 0,1 and 2 the training, validation and test number of spaces respectively
		"""
		train_size = getSizeByPorcentage(len_spaces, train_size)
		test_size = getSizeByPorcentage(len_spaces, test_size)

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


def get_subsets_cnrpark(info_filename, train_size, test_size):
		with open(info_filename, "rb") as fp:   # Unpickling
				images_info_reduced = pickle.load(fp)

		spaces = extractUniqueItemsByKey(images_info_reduced, 'space')

		spaces_distribution = asignNumberSpacesForSubsets(len(spaces), train_size, test_size)

		# For each day i would select randoms spaces for each subset (training, validation, test)
		images_info_reduced.sort(key=lambda x: x['date'])
		subsets_info = {'train': [], 'test': []}
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
				subsets_info[subset].append(image_info)
		return (data_paths, subsets_info)

def count_quantity_of_classes(data_paths):
		empty_count = 0
		occupied_count = 0
		for train_image in data_paths:
				if train_image['y'] == '0':
						empty_count += 1
				else:
						occupied_count += 1
		return [empty_count, occupied_count]


def get_paths_and_label(path, label_class, fileext='.png'):
	data_label = []
	for (dirpath, dirnames, filenames) in os.walk(path):
			data = [{'path': os.path.join(dirpath, filename), 'y': label_class} for filename in filenames if filename.endswith(fileext)]
			data_label.extend(data)
	return (data_label, len(data_label))

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

def create_subset_directorie(dir):
		subsets_path = os.path.join(base_subsets_path, dir)
		if os.path.isdir(subsets_path):
			print('This subset was already made')
			return (subsets_path, False)
		else:
			print("Make dir: {}".format(subsets_path))
			os.mkdir(subsets_path)
			return (subsets_path, True)


def save_subset(subset, subsets_dir, type):
		subsets_path = os.path.join(base_subsets_path, subsets_dir)
		toCSV = subset
		keys = toCSV[0].keys()
		with open(os.path.join(subsets_path, 'data_paths_{}.csv'.format(type)), 'w') as output_file:
				dict_writer = csv.DictWriter(output_file, delimiter=',', lineterminator='\n', fieldnames=keys)
				dict_writer.writeheader()
				dict_writer.writerows(toCSV)


def save_subsets_info(subsets, subsets_dir, extra_info=''):
		subsets_path = os.path.join(base_subsets_path, subsets_dir)
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
				info = "Subset {} size: {} from {} spaces - empty: {} ocuppied: {} - cloudy: {} sunny: {} rainy: {} \n".format(
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
		with open(os.path.join(subsets_path, 'data_info.txt'), "a") as finfo:
				finfo.write(extra_info)