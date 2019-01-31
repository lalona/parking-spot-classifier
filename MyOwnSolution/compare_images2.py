"""
La idea es simple: con una imagen donde el espacio se encuentre vacio se compara con otra imagenes y dependiendo las diferencias
se concluye si está ocupado o vacio
"""

import cv2
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
import pickle
import argparse
from operator import itemgetter
from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import ntpath
import json

path_pklot = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'


def getGrayscaleImage(filepath):
		"""
		Read the image files and converts it to gray scale
		:param filepath: the path to the image
		:return: the image in grayscale
		"""
		image = cv2.imread(filepath)
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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


def getNewImageInfo(image_info):
		image_info['filepath'] = os.path.join(path_pklot, image_info['filepath'], image_info['filename'])
		return image_info


def mse(imageA, imageB):
		# the 'Mean Squared Error' between the two images is the
		# sum of the squared difference between the two images;
		# NOTE: the two images must have the same dimension
		err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
		err /= float(imageA.shape[0] * imageA.shape[1])

		# return the MSE, the lower the error, the more "similar"
		# the two images are
		return err


def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-f", "--filename", type=str, required=True,
												help='Path to the file the contains the dictionary with the info of the dataset reduced.')

		args = vars(parser.parse_args())

		info_filename = args["filename"]

		# test set
		with open(info_filename, "rb") as fp:  # Unpickling
				images_info = pickle.load(fp)

		grouper = itemgetter('parkinglot', 'space')
		images_info = sorted(images_info, key=grouper)

		parkinglots = extractUniqueItemsByKey(images_info, 'parkinglot')

		images_info_by_patkinglot = {}

		for parkinglot in parkinglots:
				image_info_parkinglot = [i for i in images_info if i['parkinglot'] == parkinglot]
				spaces_parkinglot = extractUniqueItemsByKey(image_info_parkinglot, 'space')
				images_info_by_spaces = {}
				for space in spaces_parkinglot:
						images_info_by_spaces[space] = [getNewImageInfo(i) for i in image_info_parkinglot if i['space'] == space]
				images_info_by_patkinglot[parkinglot] = images_info_by_spaces

		# Hasta este punto ya tengo un dictionario dividido por estacionamiento que a su vez se divide por espacios

		# Voy a obtener la lista de un espacio en particular de un estacionamiento, voy a obtener el primer espacio vacio que
		# encuentre y despues voy a compararlo con los demas
		# Mostrar en una ventana el espacio vacio y en la otra la comparacion y el resultado

		empty_space_filepath = ''

		errors = []
		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				for space, images_info_of_space in images_info_by_spaces.items():
						error_count_empty = 0
						error_count_occupied = 0
						error_empty = 0
						error_occupied = 0
						empty_space_filepath = ''
						example_list = images_info_of_space
						for example in tqdm(example_list):
								if example['state'] == '0' and len(empty_space_filepath) == 0:
										empty_space_filepath = example['filepath']
										img_empty_space = getGrayscaleImage(empty_space_filepath)
										break
						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								img_comparision_space = getGrayscaleImage(comparision_space_filepath)
								try:
										sim = ssim(img_empty_space, img_comparision_space)
								except:
										height1, width1 = img_empty_space.shape
										img_comparision_space = cv2.resize(img_comparision_space, (width1, height1))
										sim = ssim(img_empty_space, img_comparision_space)
								nm = nrmse(img_empty_space, img_comparision_space)
								# m = mse(img_empty_space, img_comparision_space)
								space_comparing_name = 'state: {} sim: {} nrmse: {}'.format(example['state'], sim, nm)

								if sim < 0.4 and example['state'] == '0':
										error_count_empty += 1
										error_empty += abs(0.4 - sim)
								if sim >= 0.4 and example['state'] == '1':
										error_count_occupied += 1
										error_occupied += abs(sim - 0.4)

								if sim > 0.7:
										empty_space_filepath = example['filepath']
										img_empty_space = img_comparision_space
								"""
								fig = plt.figure('title')
								plt.suptitle(space_comparing_name)

								# show first image
								ax = fig.add_subplot(1, 2, 1)
								plt.imshow(img_empty_space, cmap=plt.cm.gray)
								plt.axis("off")

								# show the second image
								ax = fig.add_subplot(1, 2, 2)
								plt.imshow(img_comparision_space, cmap=plt.cm.gray)
								plt.axis("off")

								# show the images
								plt.show()
								"""

						error_occupied = 0 if error_count_occupied == 0 else (error_occupied / error_count_occupied)
						error_empty = 0 if error_count_empty == 0 else (error_empty / error_count_empty)
						print('In the space {} in a total of {} there was an error of occupied {} {} empty {} {}'.format(space, len(
								example_list), error_count_occupied, error_occupied, error_count_empty, error_empty))
						errors.append({'parkinglot': parkinglot, 'space': space, 'total': len(example_list),
													 'error_count_occupied': error_count_occupied,
													 'error_occupied': error_occupied,
													 'error_count_empty': error_count_empty, 'error_empty': error_empty})

		info = {'dataset': info_filename, 'threshold': 0.4, 'comparision_method': 'sim', 'errors': errors}
		dataset_name = ntpath.basename(info_filename).split('.')[0]
		feedback_filename = '{}_{}_{}.json'.format(dataset_name, 0.4, 'sim')
		with open(feedback_filename, 'w') as outfile:
				json.dump(info, outfile)


# s = ssim(grayscale_selected_image, grayscale_current_image)

if __name__ == "__main__":
		main()