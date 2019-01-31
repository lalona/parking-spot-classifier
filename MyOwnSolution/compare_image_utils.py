import pickle
from operator import itemgetter
from itertools import groupby
import os
import cv2
from PIL import Image, ImageStat
from tqdm import tqdm
from compare_images_methods2 import structural_sim, pixel_sim, sift_sim, earth_movers_distance, mse, nrmse, getMethods, get_img
import msvcrt as m

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


def getGrayscaleImage(filepath):
		"""
		Read the image files and converts it to gray scale
		:param filepath: the path to the image
		:return: the image in grayscale
		"""
		image = cv2.imread(filepath)
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def brightness(im_file):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def tryMethodsOnSpaces(images_info_by_patkinglot, database):
		dataset = {}
		methods = getMethods()
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				dataset[parkinglot] = {}
				for space, images_info_of_space in images_info_by_spaces.items():
						dataset[parkinglot][space] = []
						empty_space_filepath = ''
						example_list = images_info_of_space
						for example in tqdm(example_list):
								if example['state'] == '0' and len(empty_space_filepath) == 0:
										empty_space_filepath = example['filepath']
										empty_space_brightness = brightness(empty_space_filepath)
										empty_img = get_img(empty_space_filepath)
										height, width = empty_img.shape
										empty_img_ne = get_img(empty_space_filepath, norm_exposure=True)
										break
						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								if comparision_space_filepath == empty_space_filepath:
										continue
								comparision_space_brightness = brightness(comparision_space_filepath)
								comparision_img = get_img(comparision_space_filepath, height, width)
								comparision_img_ne = get_img(comparision_space_filepath, height, width, norm_exposure=True)
								info = {'comparing_with': empty_space_filepath, 'comparing_with_brig': empty_space_brightness,
												'state': example['state'], 'comparing_to': comparision_space_filepath, 'comparing_to_brig': comparision_space_brightness}
								for method in methods:
										try:
												if 'normal_exposure' in method:
													result = method['callback'](empty_img_ne, comparision_img_ne)
												else:
													result = method['callback'](empty_img, comparision_img)
										except:
												result = 0
										info[method['name']] = result
								dataset[parkinglot][space].append(info)
						stadistics = {}
						for info in dataset[parkinglot][space]:
								for method in methods:
										if method['name'] not in stadistics:
												stadistics[method['name']] = {'occupied': {'count': 0, 'total_result': 0},
																							'empty': {'count': 0, 'total_result': 0}}
										if info['state'] == '0':
												stadistics[method['name']]['empty']['count'] += 1
												stadistics[method['name']]['empty']['total_result'] += info[method['name']]
										else:
												stadistics[method['name']]['occupied']['count'] += 1
												stadistics[method['name']]['occupied']['total_result'] += info[method['name']]
						for method_name, stadistic in stadistics.items():
								por_result_oc = (stadistic['occupied']['total_result'] / stadistic['occupied']['count']) if \
									stadistic['occupied']['count'] > 0 else 0
								por_result_emp = (stadistic['empty']['total_result'] / stadistic['empty']['count']) if \
									stadistic['empty']['count'] > 0 else 0
								print('{} mean value oc {} emp {}'.format(method_name, por_result_oc, por_result_emp))

		dataset_name = 'dataset_{}_brightness'.format(database)
		for method in methods:
				dataset_name += '_{}'.format(method['name'])
		dataset_name += '.json'
		return (dataset, dataset_name)




