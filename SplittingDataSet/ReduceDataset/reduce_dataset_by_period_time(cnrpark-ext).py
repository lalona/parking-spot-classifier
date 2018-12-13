"""
Este programa se encarga de reducir los labels de CNRPark-EXT y almacenarlos en
un archivo .txt con el uso de pickle.
La reducción se realiza en cada espacio en un determinado día y tomado por cierta camara.
Se recorre a lo largo de las imagenes tomadas del espacio en cierto día y por cierta
camara y se toma una solo cada cierto tiempo o si el espacio cambio de estado también se
toma esa imagen.
"""
import argparse
import os
import pickle
from operator import itemgetter
from tqdm import tqdm

path_labels = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\"
path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

def main():
		parser = argparse.ArgumentParser(description='Change the period.')
		parser.add_argument("-s", "--step", type=int, required=True,
												help='This defines the step taken in the hours to take an image for each space for a respective hour and camera')
		parser.add_argument("-f", "--force", type=bool, default=False, help='Ff the file was already made, still make it')
		args = vars(parser.parse_args())

		step = args["step"]
		force_creation = args["force"]

		file_name = "cnrpark_labels_reduced_{}_{}.txt".format("period-time", step)

		if 5 < step < 0:
				print('The step has to be less than 5 and more than 0')
				return

		if os.path.isfile(file_name) and not force_creation:
				print('The file was already created if you want to repeat the process you can set the param -f to True')
				return

		images_info_file_path = "cnrpark_labels.txt"
		with (open(images_info_file_path, "rb")) as images_info_file:
				try:
					images_info = pickle.load(images_info_file)
				except EOFError:
					print('Either the path {} is wrong ot the file doesnt contain the right info'.format(images_info_file_path))
					return

		date = ''
		space = ''
		count = 0
		images_info_reduced = []
		for image_info in tqdm(images_info):
				if date != image_info['date'] or space != image_info['space']:
						# this happens when the iteration steps into a new space or date
						# the image is taken
						date = image_info['date']
						space = image_info['space']
						count = 0
				if count == 0:
						images_info_reduced.append(image_info)
						image_info_selected = image_info
						count = step
				elif image_info_selected['state'] != image_info['state']:
						images_info_reduced.append(image_info)
						image_info_selected = image_info
				count -= 1

		print("Original size: {} Reduced size: {}".format(len(images_info), len(images_info_reduced)))

		with open(file_name, "wb") as fp:  # Pickling
				pickle.dump(images_info_reduced, fp)

if __name__ == "__main__":
		main()




