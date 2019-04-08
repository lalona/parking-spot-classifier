import pickle
import os
from operator import itemgetter
import argparse
import constants.databases_info as c

def extractInfoFromLineCNRPark(label):
		"""
		Extract the information from label files and puts it in a dictionary
		:param label: It has to came in the next format:
		[weather]/[date]/[camera]/[file name] [state]
		RAINY/2016-02-12/camera1/R_2016-02-12_09.10_C01_191.jpg 1
		:return: A dictionary with the information extracted
		"""
		patch_path = label[:-3]
		patch_path_separated = patch_path.split("/")
		patch_info_separated = patch_path_separated[3].split("_")
		space = patch_info_separated[4].split(".")[0]
		state = label[-2]
		return {
				c.db_json_filepath: label[:-3],
				c.db_json_parkinglot_camera: patch_path_separated[2],
				c.db_json_weather: patch_path_separated[0],
				c.db_json_date: patch_path_separated[1],
				c.db_json_hour: patch_info_separated[2],
				c.db_json_space: space,
				c.db_json_state: state
		}


def getAllImagesInfoCNRPark(labels_files_directory):
		"""
		This will return a list of dictionares where each dictionary contains the info
		for each file
		Important consideration: it only read the camera .txt files.
		:param labels_files_directory: The path to the directory of the .txt files that contain all the image info
		:return: list of dictionaries with the image info
		"""
		# This filters the files to only open the camera files
		camera_label_files = [open(os.path.join(labels_files_directory, file_name), "r") for file_name in
													os.listdir(labels_files_directory) if file_name.startswith("camera")]
		images_info = []
		for label_file in camera_label_files:
				for label in label_file:
						images_info.append(extractInfoFromLine(label))
		return images_info

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-s", "--save-to", required=True,
										help="The dir to save the resulting file.")
		ap.add_argument("-f", "--force", required=False, default=False,
										help="Even if the files was already created for the creation.")

		args = vars(ap.parse_args())

		path_label_files = c.cnrparkext_labels_path
		save_to_dir = args['save_to']
		force = args['force']
		file_name = c.cnrparkext_labels_pickle

		if os.path.isfile(file_name):
				print('The file was already created, you can delete the file if you want to created again')
				if not force:
					return

		# this will contain all the information extracted from the labels
		images_info = getAllImagesInfoCNRPark(path_label_files)

		# sort the info
		grouper = itemgetter(c.db_json_parkinglot_camera, c.db_json_parkinglot_date, c.db_json_parkinglot_space, c.db_json_parkinglot_hour)
		images_info = sorted(images_info, key=grouper)

		print('Total of images info: {}'.format(len(images_info)))

		with open(os.path.join(save_to_dir, file_name), "wb") as fp:  # Pickling
			pickle.dump(images_info, fp)

		print("SAVED")

if __name__ == "__main__":
		main()