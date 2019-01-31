import cv2
import os
import pickle
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import argparse
from operator import itemgetter

path_images = "C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot\\"
def getGrayscaleImage(filepath):
		"""
		Read the image files and converts it to gray scale
		:param filepath: the path to the image
		:return: the image in grayscale
		"""
		image = cv2.imread(os.path.join(path_images, filepath))
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getPathToImage(image_info):
		dataset_path = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'
		path = os.path.join(dataset_path, image_info['filepath'], image_info['filename'])
		return path

def main():
	parser = argparse.ArgumentParser(description='Change the porcentage of the images.')
	parser.add_argument("-p", "--porcentage-similiraty", type=int, required=True,
												help='the porcentage of similarity aceptable, below that porcentage the image is considered different')
	parser.add_argument("-f", "--force", type=bool, default=False, help='if the file was already made, still make it')
	args = vars(parser.parse_args())

	porcentage_similarity = args["porcentage_similiraty"]
	force_creation = args["force"]

	if 100 < porcentage_similarity < 0:
			print('The porcetange similarity has to be less than 100 and more than 0')
			return

	file_name = "pklot_labels_reduced_{}_{}.txt".format("comparing-images", porcentage_similarity)

	if os.path.isfile(file_name) and not force_creation:
			print('The file was already created if you want to repeat the process you can set the param -f to True')
			return



		# The file saved in save_dataset_labels(cnrpark-ext) sort by camera,
	images_info_file_path = "pklot_labels.txt"
	with (open(images_info_file_path, "rb")) as images_info_file:
			try:
					images_info = pickle.load(images_info_file)
			except EOFError:
					print('Either the path {} is wrong ot the file doesnt contain the right info'.format(images_info_file_path))
					return

	date = ''
	space = ''
	images_info_reduced = []
	image_info_selected = images_info[0]
	for image_info in tqdm(images_info):
			if date != image_info['date'] or space != image_info['space']:
					# this happens when the iteration steps into a new space or date
					# the image is taken
					date = image_info['date']
					space = image_info['space']
					image_info_selected = image_info
					images_info_reduced.append(image_info_selected)
			elif image_info_selected['state'] != image_info['state']:
					# if the state is different from the image selected for this space in the specific date
					# then the new image selected is the current
					image_info_selected = image_info
					images_info_reduced.append(image_info_selected)
			else:
					# if is still the same space, date and state then a comparision is made, if this comparicion
					# is less than the porcentage similarity then the current image is selected
					grayscale_selected_image = getGrayscaleImage(getPathToImage(image_info_selected))
					grayscale_current_image = getGrayscaleImage(getPathToImage(image_info))
					s = ssim(grayscale_selected_image, grayscale_current_image)
					if (s * 100) < porcentage_similarity:
						image_info_selected = image_info
						images_info_reduced.append(image_info_selected)

	print("Original size: {} Reduced size: {}".format(len(images_info), len(images_info_reduced)))

	with open(file_name, "wb") as fp:  # Pickling
			pickle.dump(images_info_reduced, fp)


if __name__ == "__main__":
		main()



