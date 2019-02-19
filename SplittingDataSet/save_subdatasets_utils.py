from itertools import groupby
import os
import csv
from random import shuffle
from tqdm import tqdm
import pickle
from operator import itemgetter

base_subsets_path = "C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\Train\\subsets"
path_pklot = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'
path_cnrpark = 'C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES'

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

def getImagePathPklot(image_info):
	return os.path.join(path_pklot, image_info['filepath'], image_info['filename'])

def getImagePathCnrpark(image_info):
	return os.path.join(path_cnrpark, image_info['filePath'])

#def get_subsets_cnrpark(info_filename, train_size, test_size):
#		return getSubsets(info_filename, train_size, test_size, getImagePathCnrpark)

def getSubsets(info_filename, train_size, test_size, database):
		getImagePath = getImagePathCnrpark if database == 'cnrpark' else getImagePathPklot
		with open(info_filename, "rb") as fp:  # Unpickling
				images_info_reduced = pickle.load(fp)

		spaces = extractUniqueItemsByKey(images_info_reduced, 'space')

		spaces_distribution = asignNumberSpacesForSubsets(len(spaces), train_size, test_size)

		# For each day i would select randoms spaces for each subset (training, validation)
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
				image_path = getImagePath(image_info)
				data_path_state = {'path': image_path, 'y': image_info['state']}
				data_paths[subset].append(data_path_state)
				subsets_info[subset].append(image_info)

		return (data_paths, subsets_info)


def getSubsets_TestOnly(info_filename, database):
		getImagePath = getImagePathCnrpark if database == 'cnrpark' else getImagePathPklot
		with open(info_filename, "rb") as fp:  # Unpickling
				images_info_reduced = pickle.load(fp)

		# For each day i would select randoms spaces for each subset (training, validation)
		images_info_reduced.sort(key=lambda x: x['date'])
		data_paths = []
		subsets_info = {'train': [], 'test': []}

		for image_info in tqdm(images_info_reduced):
				image_path = getImagePath(image_info)
				data_path_state = {'path': image_path, 'y': image_info['state']}
				data_paths.append(data_path_state)
				subsets_info['test'].append(image_info)

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

def getPathsAndLabelDataaug(path, fileext='.png'):
	data_label = []
	for (dirpath, dirnames, filenames) in os.walk(path):
			data = [{'path': os.path.join(dirpath, filename), 'y': filename[0]} for filename in filenames if filename.endswith(fileext)]
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
		# esto se hace porque causa muchos problemas que el primer estado sea diferente en los diferetnes subconjuntos train and test
		first_state = subset[0]['y']
		if first_state == '1':
				i = 0
				for data in subset:
						if data['y'] == '0':
								break
						i += 1
				subset[i], subset[0] = subset[0], subset[i]

		subsets_path = os.path.join(base_subsets_path, subsets_dir)
		toCSV = subset
		keys = toCSV[0].keys()
		with open(os.path.join(subsets_path, 'data_paths_{}.csv'.format(type)), 'w') as output_file:
				dict_writer = csv.DictWriter(output_file, delimiter=',', lineterminator='\n', fieldnames=keys)
				dict_writer.writeheader()
				dict_writer.writerows(toCSV)

def saveSubsetsInfo(subsets, subsets_dir, database, extra_info=''):
		if database == 'cnrpark':
				save_subsets_info_cnrpark(subsets, subsets_dir, extra_info)
		else:
				save_subsets_info_pklot(subsets, subsets_dir, extra_info)

def save_subsets_info_cnrpark(subsets, subsets_dir, extra_info=''):
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


def save_subsets_info_pklot(subsets, subsets_dir, extra_info=''):
		subsets_path = os.path.join(base_subsets_path, subsets_dir)
		for key, value in subsets.items():
				spacesIn = extractUniqueItemsByKey(value, 'space')
				empty_count = 0
				ocuppied_count = 0
				cloudy_count = 0
				sunny_count = 0
				rainy_count = 0
				for v in value:
						if v['state'] == '0':
								empty_count += 1
						else:
								ocuppied_count += 1
						if v['weather'] == 'Cloudy':
							cloudy_count += 1
						elif v['weather'] == 'Sunny':
							sunny_count += 1
						elif v['weather'] == 'Rainy':
							rainy_count += 1
				info = "Subset {} size: {} from {} spaces - empty: {} ocuppied: {} - cloudy: {} sunny: {} rainy: {} \n".format(
						key,
						len(
								value),
						len(
								spacesIn),
						empty_count,
						ocuppied_count,
						cloudy_count,
						sunny_count,
						rainy_count)
				print(info)
				with open(os.path.join(subsets_path, 'data_info.txt'), "a") as finfo:
						finfo.write(info)
		with open(os.path.join(subsets_path, 'data_info.txt'), "a") as finfo:
				finfo.write(extra_info)