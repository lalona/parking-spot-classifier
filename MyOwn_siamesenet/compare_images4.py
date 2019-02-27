"""
La idea es simple: con una imagen donde el espacio se encuentre vacio se compara con otra imagenes y dependiendo las diferencias
se concluye si está ocupado o vacio
Las pruebas van a tomar la primer imagen de un espacio vacio que exista dentro de todas las imagenes pertenecientes a un espacio
en concreto en un estacionamiento en concreto, esta imagen va a ser comparada usa dos metodos ssim y nrmse pero se puede hacer uso de cualquier otro
Aqui se obtienen varios valores al comparar imagenes, dependiendo de los metodos que se usen en campare_images_methods2
Debe funcionar con cnrpark
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
from compare_image_utils import get_images_info_by_parkinglot, getDataset, getDataset2, getDatasetCar
import ntpath
path_pklot = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'

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
		parser.add_argument("-d", "--database", type=str, required=True,
												help='Name of dtaabase can be cnpark or pklot.')

		args = vars(parser.parse_args())

		info_filename = args["filename"]
		database = args["database"]

		specific_database = ntpath.basename(info_filename)
		specific_database = os.path.splitext(specific_database)[0]

		if database == 'cnrpark':
			images_info_by_patkinglot = get_images_info_by_parkinglot(info_filename, parkinglot_key='camera', database=database)
		else:
			images_info_by_patkinglot = get_images_info_by_parkinglot(info_filename, parkinglot_key='parkinglot',
																																	database=database)
		# Hasta este punto ya tengo un dictionario dividido por estacionamiento que a su vez se divide por espacios

		# Voy a obtener la lista de un espacio en particular de un estacionamiento, voy a obtener el primer espacio vacio que
		# encuentre y despues voy a compararlo con los demas
		# Mostrar en una ventana el espacio vacio y en la otra la comparacion y el resultado


		dataset, dataset_name = getDatasetCar(images_info_by_patkinglot, specific_database=specific_database)

		with open(dataset_name, 'w') as outfile:
				json.dump(dataset, outfile)


# s = ssim(grayscale_selected_image, grayscale_current_image)

if __name__ == "__main__":
		main()