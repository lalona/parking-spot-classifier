import pickle
import os
from operator import itemgetter

path_labels = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\"

def extractInfoFromLine(label):
		"""
		Extract the information from label files and puts it in a disctionary
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
				'filePath': label[:-3],
				'camera': patch_path_separated[2],
				'weather': patch_path_separated[0],
				'date': patch_path_separated[1],
				'hour': patch_info_separated[2],
				'space': space,
				'state': state
		}


def getAllImagesInfo(labels_files_directory):
		"""
		This will return a list of dictionares where each dictionary contains the info
		for each file
		Important consideration: it only read the camera .txt files.
		:param labels_files_directory: The path to the directory of the .txt files that contain all the image info
		:return: list of dictionaries with the image info
		"""
		path_label_files = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\"
		# This filters the files to only open the camera files
		camera_label_files = [open(os.path.join(labels_files_directory, file_name), "r") for file_name in
													os.listdir(labels_files_directory) if file_name.startswith("camera")]
		images_info = []
		for label_file in camera_label_files:
				for label in label_file:
						images_info.append(extractInfoFromLine(label))
		return images_info


def sortByKeys(list, *keys):
		grouper = itemgetter(keys)
		return sorted(list, key=grouper)

def main():
		file_name = "cnrpark_labels.txt"

		if os.path.isfile(file_name):
				print('The file was already created, you can delete the file if you want to created again')
				return

		# this will contain all the information extracted from the labels
		images_info = getAllImagesInfo(path_labels)

		# sort the info
		grouper = itemgetter('camera', 'date', 'space', 'hour')
		images_info = sorted(images_info, key=grouper)

		print('Total of images info: {}'.format(len(images_info)))

		with open(file_name, "wb") as fp:  # Pickling
				pickle.dump(images_info, fp)

		print("SAVED")

if __name__ == "__main__":
		main()