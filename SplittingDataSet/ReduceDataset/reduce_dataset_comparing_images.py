"""
SPNISH
En este programa se reduce el dataset comparando las imagenes. Por cada espacio en determinado día en determinada
camara se itera a lo largo de las horas, se toma la imagen del espacio en la primer hora, la imagen que se toma
se compara con la siguiente y si el parecido de acurdo a la formula ssim da menos que cierto porcentaje definido
por el usuario entonces se toma la imagen y es la nueva imagen con la que se compara,
tambien si el espacio cambio de estado se toma la imagen.

El programa guarda los resultados de reducir las imagenes en un archivo .txt usando el protocolo pickle de python,
el nombre del archico sigue el siguiente formato:
	cnrpark_labels_reduced_[method]_[porcetange].txt
el metodo en este caso es 'comparing_images' y 'procentage' representa el porcentaje definido por el usuario
"""

"""
ENGLISH 
This reduces the dataset comparing the images. For space in a specific date and camera the images are reduced:
selecting the first image taked in the first hour of that day, the first image selected is compared with
the image taken in the next hour using the ssim formula, if the result of the comparision is less than a porcentage
insserted by the user the image is taken or if the state of the space has chance the image is taken, the new image taken 
is now the one that is used to compare the next images.

"""
import argparse
from operator import itemgetter
from tqdm import tqdm
import os
import cv2
from skimage.measure import compare_ssim as ssim
import pickle
import constants.databases_info as c

def getGrayscaleImage(filepath):
		"""
		Read the image files and converts it to gray scale
		:param filepath: the path to the image
		:return: the image in grayscale
		"""
		image = cv2.imread(filepath)
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getImagesInfoReduced(database, porcentage_similarity, labels_dir):
		# The file saved in save_dataset_labels(cnrpark-ext) sort by camera,
		if database == 'cnrpark':
			images_info_file_path = os.path.join(labels_dir, c.cnrparkext_labels_pickle)
			images_path = c.cnrparkext_patches_path
		elif database == 'pklot':
			images_info_file_path = os.path.join(labels_dir, c.pklot_labels_pickle)
			images_path = c.pklot_patches_path

		with (open(images_info_file_path, "rb")) as images_info_file:
				try:
						images_info = pickle.load(images_info_file)
				except EOFError:
						print('Either the path {} is wrong or the file doesnt contain the right info'.format(images_info_file_path))
						return

		grouper = itemgetter(c.db_json_parkinglot_camera, c.db_json_date, c.db_json_space,
												 c.db_json_hour)
		images_info = sorted(images_info, key=grouper)

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
						path_image_selected = os.path.join(images_path, image_info_selected[c.db_json_filepath])
						path_image = os.path.join(images_path, image_info[c.db_json_filepath])
						grayscale_selected_image = getGrayscaleImage(path_image_selected)
						grayscale_current_image = getGrayscaleImage(path_image)
						s = ssim(grayscale_selected_image, grayscale_current_image)
						if (s * 100) < porcentage_similarity:
								image_info_selected = image_info
								images_info_reduced.append(image_info_selected)
						print('{} vs {}'.format(image_info_selected['filePath'], image_info['filePath']))
						print('result: {}'.format(s * 100))
		print("Original size: {} Reduced size: {}".format(len(images_info), len(images_info_reduced)))
		return images_info_reduced


def main():
		parser = argparse.ArgumentParser(description='Change the porcentage of the images.')
		parser.add_argument("-d", "--database", required=True,
												help="It can be cnrpark or pklot.")
		parser.add_argument("-l", "--labels-dir", required=True,
												help="The dir where the labels where saved.")
		parser.add_argument("-s", "--save-to", required=True,
												help="The dir to save the resulting file.")
		parser.add_argument("-p", "--porcentage-similiraty", type=int, required=True,
												help='the porcentage of similarity aceptable, below that porcentage the image is considered different')
		parser.add_argument("-f", "--force", type=bool, default=False, help='if the file was already made, still make it')
		args = vars(parser.parse_args())

		database = args['database']
		if database != 'cnrpark' or database != 'pklot':
				print('The database it can be only cnrpark or pklot')
				return
		save_to_dir = args['save_to']
		porcentage_similarity = args["porcentage_similiraty"]
		force_creation = args["force"]
		labels_dir = args['labels_dir']

		file_name = "{}_labels_reduced_{}_{}.txt".format(database, "comparing-images", porcentage_similarity)
		file_name = os.path.join(save_to_dir, file_name)

		if 100 < porcentage_similarity < 0:
				print('The porcetange similarity has to be less than 100 and more than 0')
				return

		if os.path.isfile(file_name) and not force_creation:
				print('The file was already created if you want to repeat the process you can set the param -f to True')
				return

		images_info_reduced = getImagesInfoReduced(database, porcentage_similarity, labels_dir)

		with open(file_name, "wb") as fp:  # Pickling
				pickle.dump(images_info_reduced, fp)


if __name__ == "__main__":
		main()
