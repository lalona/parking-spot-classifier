import argparse
import json
from tqdm import tqdm
import os
import cv2
import imutils
import msvcrt as m
import operator
import pickle

"""
Primer objetivo: agrupar las imagenes por estacionamiento
Segundo objetivo: poder preguntar por los espacios en los que más se equivoco por estacionamiento v
Tercer objetivo: comparar el numero de imagenes con error con el numero real de imagenes
"""

def main():
	print('hola')
	ap = argparse.ArgumentParser()
	ap.add_argument('-e', '--error-file', type=str, required=True, help='The path to the file that contains the error images')


	args = vars(ap.parse_args())

	error_file = args['error_file']

	with open(error_file, 'r') as fjson:
			failed_images = json.load(fjson)

	with open(failed_images['dataset'], "rb") as fp:  # Unpickling
			all_images_info = pickle.load(fp)

	stadistics = {}

	for key, value in failed_images['image_info'][0].items():
		stadistics[key] = {}

	high_error_images = []
	# agrupa las imagenes por estacionamiento
	failed_images_by_parking = {}

	for failed_image in tqdm(failed_images['image_info']):
			parking = failed_image['parkinglot']
			if parking in failed_images_by_parking:
					failed_images_by_parking[parking].append(failed_image)
			else:
					failed_images_by_parking[parking] = []

	images_info_by_parking = {}

	for image_info in tqdm(all_images_info):
			parking = image_info['parkinglot']
			if parking in images_info_by_parking:
					images_info_by_parking[parking].append(image_info)
			else:
					images_info_by_parking[parking] = []

	# por cada estacionamiento hace un diccionario con el espacio como key y el numero de espacios como value
	parkinglot_spaces = {}
	for parkinglot, failed_images in failed_images_by_parking.items():
			print(parkinglot)
			parkinglot_spaces[parkinglot] = {}
			for failed_image in failed_images:
					space = failed_image['space']
					if space in parkinglot_spaces[parkinglot]:
							parkinglot_spaces[parkinglot][space] += 1
					else:
							parkinglot_spaces[parkinglot][space] = 1

	image_info_parkinglot_spaces = {}
	for parkinglot, images_info in images_info_by_parking.items():
			print(parkinglot)
			image_info_parkinglot_spaces[parkinglot] = {}
			for image_info in images_info:
					space = image_info['space']
					if space in image_info_parkinglot_spaces[parkinglot]:
							image_info_parkinglot_spaces[parkinglot][space] += 1
					else:
							image_info_parkinglot_spaces[parkinglot][space] = 1

	top = -1
	low_porcentage = 30
	while(True):
		c = ord(m.getch())
		if c == ord('s'):
				for parkinglot, failed_spaces in parkinglot_spaces.items():
					sorted_by_value = sorted(failed_spaces.items(), key=operator.itemgetter(1))
					print("For the parking lot {} the top {} spaces with error are:".format(parkinglot, top))
					count = 0
					spaces_low_porcentage_error = 0
					start_porcetange = 60
					for space, value in reversed(sorted_by_value):

						if count > top:
								if top > -1: break
						total_space_images = image_info_parkinglot_spaces[parkinglot][space]
						error_porcentage = int((value * 100) / total_space_images)
						if error_porcentage <= start_porcetange:
								print("Below this the error is equal or less than {}".format(start_porcetange))
								start_porcetange -= 10
						if error_porcentage <= low_porcentage:
								spaces_low_porcentage_error += 1
						print("space {}: {} - {}".format(space, value, total_space_images))
						count += 1
					print("From a total of {} spaces {} has a {} or less porcentage error".format(len(sorted_by_value), spaces_low_porcentage_error, low_porcentage))
		if c == ord('b'):
				break

if __name__ == "__main__":
		main()