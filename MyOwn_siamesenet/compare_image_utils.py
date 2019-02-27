import pickle
from operator import itemgetter
from itertools import groupby
import os
import cv2
from PIL import Image, ImageStat
from tqdm import tqdm
import msvcrt as m
import random

path_pklot = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'
path_cnrpark = 'C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES'
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


def getNewImageInfo(image_info, database='pklot'):
		if database == 'pklot':
			image_info['filepath'] = os.path.join(path_pklot, image_info['filepath'], image_info['filename'])
		else:
				image_info['filepath'] = os.path.join(path_cnrpark, image_info['filePath'])
		return image_info

def get_images_info_by_parkinglot(info_filename, parkinglot_key='parkinglot', database='pklot'):
		# Regresa un dictionario dividido por estacionamiento que a su vez se divide por espacios
		# test set
		with open(info_filename, "rb") as fp:  # Unpickling
			images_info = pickle.load(fp)

		grouper = itemgetter(parkinglot_key, 'space')
		images_info = sorted(images_info, key=grouper)

		parkinglots = extractUniqueItemsByKey(images_info, parkinglot_key)

		images_info_by_patkinglot = {}

		for parkinglot in parkinglots:
				image_info_parkinglot = [i for i in images_info if i[parkinglot_key] == parkinglot]
				spaces_parkinglot = extractUniqueItemsByKey(image_info_parkinglot, 'space')
				images_info_by_spaces = {}
				for space in spaces_parkinglot:
						images_info_by_spaces[space] = [getNewImageInfo(i, database) for i in image_info_parkinglot if i['space'] == space]
				images_info_by_patkinglot[parkinglot] = images_info_by_spaces
		return images_info_by_patkinglot


def getImagesInfo(info_filename, database='pklot'):
		# Regresa un dictionario dividido por estacionamiento que a su vez se divide por espacios
		# test set
		with open(info_filename, "rb") as fp:  # Unpickling
				images_info = pickle.load(fp)
		return [getNewImageInfo(i, database) for i in images_info]


def getDataset(images_info_by_patkinglot, specific_database):
		"""
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						example_list = images_info_of_space
						for example in tqdm(example_list):
								if example['state'] == '0' and len(empty_space_filepath) == 0:
										empty_space_filepath = example['filepath']
										break
						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								#if comparision_space_filepath == empty_space_filepath:
								#		continue
								info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath, 'state': example['state']}
								dataset[parkinglot][space].append(info)
		dataset_name = 'dataset_{}'.format(specific_database)
		dataset_name += '.json'
		return (dataset, dataset_name)


def getDatasetCar(images_info_by_patkinglot, specific_database):
		"""
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						example_list = images_info_of_space
						for example in tqdm(example_list):
								if example['state'] == '1' and len(empty_space_filepath) == 0:
										empty_space_filepath = example['filepath']
										break
						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								# if comparision_space_filepath == empty_space_filepath:
								#		continue
								info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
												'state': example['state']}
								dataset[parkinglot][space].append(info)
		dataset_name = 'dataset_compoccupied_{}'.format(specific_database)
		dataset_name += '.json'
		return (dataset, dataset_name)


def getDataset_ForTests1(images_info_by_patkinglot, specific_database, empty_spaces=1):
		"""
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						empty_spaces_comparision = []
						example_list = images_info_of_space
						random.shuffle(example_list)
						for example in tqdm(example_list):
								if example['state'] == '1':
										continue
								if len(empty_spaces_comparision) >= empty_spaces:
										break
								empty_spaces_comparision.append(example['filepath'])

						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								if comparision_space_filepath in empty_spaces_comparision:
										continue
								# if comparision_space_filepath == empty_space_filepath:
								#		continue
								info = {'comparing_with': empty_spaces_comparision, 'comparing_to': comparision_space_filepath,
												'state': example['state']}
								dataset[parkinglot][space].append(info)
		dataset_name = 'dataset_{}_for_test1_{}es'.format(specific_database, empty_spaces)
		dataset_name += '.json'
		return (dataset, dataset_name)


def getDataset2(images_info_by_patkinglot, database, specific_database, empty_spaces_by_space, half_total_comparisions):
		"""
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						random.shuffle(images_info_of_space)
						comparision_with_list = []
						for example in tqdm(images_info_of_space):
								if example['state'] == '0' and len(comparision_with_list) < empty_spaces_by_space:
										empty_space_filepath = example['filepath']
										comparision_with_list.append(empty_space_filepath)
						print(len(comparision_with_list))
						for empty_space_filepath in comparision_with_list:
								random.shuffle(images_info_of_space)
								negative_comparisions = 0
								positive_comparisions = 0
								for example in tqdm(images_info_of_space):
										if example['state'] == '0':
												if positive_comparisions >= half_total_comparisions:
														continue
												positive_comparisions += 1
										else:
												if negative_comparisions >= half_total_comparisions:
														continue
												negative_comparisions += 1
										comparision_space_filepath = example['filepath']
										info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
														'state': example['state']}
										dataset[parkinglot][space].append(info)
								#print('{} {}'.format(negative_comparisions, positive_comparisions))

		dataset_name = 'dataset_{}_{}s_{}hc'.format(specific_database, empty_spaces_by_space, half_total_comparisions)
		dataset_name += '.json'
		return (dataset, dataset_name)


def getDataset3(images_info_by_patkinglot, specific_database, empty_spaces_by_space, half_total_comparisions):
		"""
		Ya no se repiten
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						random.shuffle(images_info_of_space)
						comparision_with_list = []
						for example in tqdm(images_info_of_space):
								if example['state'] == '0' and len(comparision_with_list) < empty_spaces_by_space:
										empty_space_filepath = example['filepath']
										comparision_with_list.append(empty_space_filepath)
						already_used = []

						for empty_space_filepath in comparision_with_list:
								random.shuffle(images_info_of_space)
								negative_comparisions = 0
								positive_comparisions = 0
								dataset_info = []
								for example in tqdm(images_info_of_space):
										if example in already_used:
												continue
										if example['state'] == '0':
												if positive_comparisions >= half_total_comparisions:
														continue
												positive_comparisions += 1
										else:
												if negative_comparisions >= half_total_comparisions:
														continue
												negative_comparisions += 1
										already_used.append(example)
										comparision_space_filepath = example['filepath']
										info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
														'state': example['state']}
										#dataset[parkinglot][space].append(info)
										dataset_info.append(info)
								dataset[parkinglot][space].extend(dataset_info)


		dataset_name = 'dataset_{}_{}s_{}hc_v2'.format(specific_database, empty_spaces_by_space, half_total_comparisions)
		dataset_name += '.json'
		return (dataset, dataset_name)


def getDataset4(images_info_by_patkinglot, images_info, specific_database, empty_spaces_by_space, half_total_comparisions):
	"""
	Ya no va a ser exlusivamente con imagenes del mismo espacio
	:param images_info_by_patkinglot: the raw info of
	:param database: pklot or cnrpark
	:return:
	"""
	dataset = {}

	for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
			dataset[parkinglot] = {}
			for space, images_info_of_space in images_info_by_spaces.items():
					dataset[parkinglot][space] = []
					random.shuffle(images_info_of_space)
					comparision_with_list = []
					for example in tqdm(images_info_of_space):
							if example['state'] == '0' and len(comparision_with_list) < empty_spaces_by_space:
									empty_space_filepath = example['filepath']
									if empty_space_filepath not in comparision_with_list:
										comparision_with_list.append(empty_space_filepath)
					for empty_space_filepath in comparision_with_list:
							random.shuffle(images_info_of_space)
							negative_comparisions = 0
							positive_comparisions = 0
							dataset_info = []
							random.shuffle(images_info)
							already_used = []
							for example in tqdm(images_info):
									if example['state'] == '0':
											if positive_comparisions >= half_total_comparisions:
													if negative_comparisions >= half_total_comparisions:
														break
													else:
														continue
											positive_comparisions += 1
									else:
											if negative_comparisions >= half_total_comparisions:
													if positive_comparisions >= half_total_comparisions:
														break
													else:
														continue
											negative_comparisions += 1
									already_used.append(example)
									comparision_space_filepath = example['filepath']
									info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
													'state': example['state']}
									# dataset[parkinglot][space].append(info)
									dataset_info.append(info)
							images_info = [img_info for img_info in images_info if img_info not in already_used]
							dataset[parkinglot][space].extend(dataset_info)

	dataset_name = 'dataset_{}_{}s_{}hc_v3'.format(specific_database, empty_spaces_by_space, half_total_comparisions)
	dataset_name += '.json'
	return (dataset, dataset_name)

def getDataset4_1(images_info, database, specific_database, compare_to=2):
	"""
	Los espacios vacios se van a comparar con n espacios aleatorios vacios y con n espacios aleatorios
	ocupados
	:param images_info:
	:param specific_database:
	:param compare_to:
	:return:
	"""
	dataset = {}
	images_info_empty = []
	images_info_occupied = []
	for image_info in images_info:
			if image_info['state'] == '0':
					images_info_empty.append(image_info)
			else:
					images_info_occupied.append(image_info)

	parkinglot_key = 'camera' if database == 'cnrpark' else 'parkinglot'
	for image_info_empty in images_info_empty:
			compare_to_positive = []
			compare_to_negative = []
			while len(compare_to_positive) < compare_to:
					p = random.choice(images_info_empty)
					if p not in compare_to_positive:
							compare_to_positive.append(p)
			while len(compare_to_negative) < compare_to:
					n = random.choice(images_info_occupied)
					if n not in compare_to_negative:
							compare_to_negative.append(n)
			parkinglot = image_info_empty[parkinglot_key]
			space = image_info_empty['space']
			if parkinglot not in dataset:
					dataset[parkinglot] = {}
			if space not in dataset[parkinglot]:
					dataset[parkinglot][space] = []
			image_info_empty_filepath = getNewImageInfo(image_info_empty, database)['filepath']
			for p in compare_to_positive:
					info = {'comparing_with': image_info_empty_filepath,
									'comparing_to': getNewImageInfo(p, database)['filepath'],
									'state': '1'}
					dataset[parkinglot][space].append(info)
			for n in compare_to_negative:
					info = {'comparing_with': image_info_empty_filepath,
									'comparing_to': getNewImageInfo(n, database)['filepath'],
									'state': '0'}
					dataset[parkinglot][space].append(info)

	dataset_name = 'dataset_{}_{}ct_v3_1'.format(specific_database, compare_to)
	dataset_name += '.json'
	print(len(dataset))
	return (dataset, dataset_name)


def getDataset5(images_info_by_patkinglot, images_info, specific_database, empty_spaces_by_space, half_total_comparisions):
	"""
	Ya no va a ser exlusivamente con imagenes del mismo espacio
	:param images_info_by_patkinglot: the raw info of
	:param database: pklot or cnrpark
	:return:
	"""
	dataset = {}

	for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
			dataset[parkinglot] = {}
			for space, images_info_of_space in images_info_by_spaces.items():
					dataset[parkinglot][space] = []
					random.shuffle(images_info_of_space)
					comparision_with_list = []
					for example in tqdm(images_info_of_space):
							if example['state'] == '0' and len(comparision_with_list) < empty_spaces_by_space:
									empty_space_filepath = example['filepath']
									if empty_space_filepath not in comparision_with_list:
										comparision_with_list.append(empty_space_filepath)
					for empty_space_filepath in comparision_with_list:
							random.shuffle(images_info_of_space)
							negative_comparisions = 0
							positive_comparisions = 0
							dataset_info = []
							random.shuffle(images_info)
							already_used = []
							for example in tqdm(images_info_of_space):
									if example['state'] == '0':
											if positive_comparisions >= half_total_comparisions:
													if negative_comparisions >= half_total_comparisions:
														break
													else:
														continue
											positive_comparisions += 1
									else:
											if negative_comparisions >= half_total_comparisions:
													if positive_comparisions >= half_total_comparisions:
														break
													else:
														continue
											negative_comparisions += 1

									comparision_space_filepath = example['filepath']
									info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
													'state': example['state']}
									# dataset[parkinglot][space].append(info)
									dataset_info.append(info)
							for example in tqdm(images_info):
									if example['state'] == '0':
											if positive_comparisions >= half_total_comparisions:
													if negative_comparisions >= half_total_comparisions:
															break
													else:
															continue
											positive_comparisions += 1
									else:
											if negative_comparisions >= half_total_comparisions:
													if positive_comparisions >= half_total_comparisions:
															break
													else:
															continue
											negative_comparisions += 1
									already_used.append(example)
									comparision_space_filepath = example['filepath']
									info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath,
													'state': example['state']}
									# dataset[parkinglot][space].append(info)
									dataset_info.append(info)

							images_info = [img_info for img_info in images_info if img_info not in already_used]
							dataset[parkinglot][space].extend(dataset_info)

	dataset_name = 'dataset_{}_{}s_{}hc_v4'.format(specific_database, empty_spaces_by_space, half_total_comparisions)
	dataset_name += '.json'
	return (dataset, dataset_name)

def getDataset6(images_info_by_patkinglot, images_info, specific_database, empty_spaces_by_space, half_total_comparisions):
	"""
	Ya no va a ser exlusivamente con imagenes del mismo espacio
	Tambien la imagen de comparing with va a tener carros
	:param images_info_by_patkinglot: the raw info of
	:param database: pklot or cnrpark
	:return:
	"""
	dataset = {}

	for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
			dataset[parkinglot] = {}
			for space, images_info_of_space in images_info_by_spaces.items():
					dataset[parkinglot][space] = []
					random.shuffle(images_info_of_space)
					comparision_with_list_empty = []
					comparision_with_list_occupied = []
					for example in tqdm(images_info_of_space):
							if example['state'] == '0' and len(comparision_with_list_empty) < empty_spaces_by_space:
									empty_space_filepath = example['filepath']
									if empty_space_filepath not in comparision_with_list_empty:
											comparision_with_list_empty.append(empty_space_filepath)
					for example in tqdm(images_info_of_space):
							if example['state'] == '1' and len(comparision_with_list_occupied) < empty_spaces_by_space:
									empty_space_filepath = example['filepath']
									if empty_space_filepath not in comparision_with_list_occupied:
											comparision_with_list_occupied.append(empty_space_filepath)
					for empty_space_filepath in comparision_with_list_empty:
							dataset_info, already_used = getDatasetInfoBySpace(empty_space_filepath, images_info_of_space, images_info, half_total_comparisions)
							images_info = [img_info for img_info in images_info if img_info not in already_used]
							dataset[parkinglot][space].extend(dataset_info)
					for empty_space_filepath in comparision_with_list_occupied:
							dataset_info, already_used = getDatasetInfoBySpace(empty_space_filepath, images_info_of_space, images_info, half_total_comparisions, positive_state='1')
							images_info = [img_info for img_info in images_info if img_info not in already_used]
							dataset[parkinglot][space].extend(dataset_info)

	dataset_name = 'dataset_{}_{}s_{}hc_v5'.format(specific_database, empty_spaces_by_space, half_total_comparisions)
	dataset_name += '.json'
	return (dataset, dataset_name)

def getDatasetInfoBySpace(empty_space_filepath, images_info_of_space, images_info, half_total_comparisions, positive_state='0'):
		random.shuffle(images_info_of_space)
		negative_comparisions = 0
		positive_comparisions = 0
		dataset_info = []
		random.shuffle(images_info)
		already_used = []
		for example in tqdm(images_info_of_space):

				if example['state'] == positive_state:
						if positive_comparisions >= half_total_comparisions:
								if negative_comparisions >= half_total_comparisions:
										break
								else:
										continue
						positive_comparisions += 1
						state = '1'
				else:
						if negative_comparisions >= half_total_comparisions:
								if positive_comparisions >= half_total_comparisions:
										break
								else:
										continue
						negative_comparisions += 1
						state = '0'

				comparision_space_filepath = example['filepath']
				info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath, 'state': state}
				# dataset[parkinglot][space].append(info)
				dataset_info.append(info)
		for example in tqdm(images_info):
				if example['state'] == positive_state:
						if positive_comparisions >= half_total_comparisions:
								if negative_comparisions >= half_total_comparisions:
										break
								else:
										continue
						positive_comparisions += 1
						state = '1'
				else:
						if negative_comparisions >= half_total_comparisions:
								if positive_comparisions >= half_total_comparisions:
										break
								else:
										continue
						negative_comparisions += 1
						state = '0'
				already_used.append(example)
				comparision_space_filepath = example['filepath']
				info = {'comparing_with': empty_space_filepath, 'comparing_to': comparision_space_filepath, 'state': state}
				# dataset[parkinglot][space].append(info)
				dataset_info.append(info)
		return (dataset_info, already_used)


def getDataset7(images_info_by_patkinglot, specific_database):
		"""
		De cada espacio se compara con un espacio vacio y con uno ocupado
		:param images_info_by_patkinglot: the raw info of
		:param database: pklot or cnrpark
		:return:
		"""
		dataset = {}
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in tqdm(images_info_by_spaces.items()):
						dataset[parkinglot][space] = []
						dataset_info = []

						images_info_of_space_empty = []
						images_info_of_space_occupied = []
						for space_ in images_info_of_space:
							if space_['state'] == '0':
									images_info_of_space_empty.append(space_)
							else:
									images_info_of_space_occupied.append(space_)
						for space_ in images_info_of_space:
								comparison_to_empty = random.choice(images_info_of_space_empty)
								comparison_to_occupied = random.choice(images_info_of_space_occupied)
								if space_['state'] == '0':
										comparison_to_empty_state = '1'
										comparison_to_occupied_state = '0'
								else:
										comparison_to_empty_state = '0'
										comparison_to_occupied_state = '1'
								info1 = {'comparing_with': space_['filepath'], 'comparing_with_state': space_['state'],
												'comparing_to': comparison_to_empty['filepath'], 'state': comparison_to_empty_state}
								info2 = {'comparing_with': space_['filepath'], 'comparing_with_state': space_['state'],
												'comparing_to': comparison_to_occupied['filepath'], 'state': comparison_to_occupied_state}
								dataset_info.append(info1)
								dataset_info.append(info2)
						dataset[parkinglot][space].extend(dataset_info)
		dataset_name = 'dataset_{}_v6'.format(specific_database)
		dataset_name += '.json'
		return (dataset, dataset_name)