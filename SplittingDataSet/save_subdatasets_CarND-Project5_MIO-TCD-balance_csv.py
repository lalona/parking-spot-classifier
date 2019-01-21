"""
Voy a tomar al azar imagenes de los conjuntos de entrenamiento CarND-Project5 y de MIO-TCD

De CarND voy a tomar las imagenes positivas de la carpeta vehicles (8,792) y las imagenes negativas
de la carpeta non-vehicles (8,970)

De MIO-TCD voy a tomar las imagenes positivas de la carpeta train\cars (260,518) y las imagenes negativas
de la carpeta train\background (160,000)

Para el primer experimento el conjunto de entrenamiento va a consistir de:
	Para los carros: todas la imagenes que hay en CarND (8,798) y 9,202 imagenes agarradas al azar
	de MIO-TCD
	Para los ejemplos negativos: todas las imagenes que hay en CarND (8,970) y 9,030 imagenes escogidas
	al azar de MIO-TCD

Para el conjunto de entrenamiento voy a escoger un conjunto de datos balanceado de
(3,000) imagenes del estacionamiento CNRPark-EXT de

"""

import os
import random
import pickle
from save_subdatasets_utils import save_subset, save_subsets_info, create_subset_directorie, get_paths_and_label
import ntpath

path_carnd_vehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\CarND-Project5\\vehicles'
path_carnd_nonvehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\CarND-Project5\\non-vehicles'

path_miotcd_vehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\MIO-TCD\\train\\car'
path_miotcd_nonvehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\MIO-TCD\\train\\background'

info_filename = 'C:\\Eduardo\\ProyectoFinal\\Proyecto\\ProyectoFinal\\SplittingDataSet\\ReduceDataset\\cnrpark_labels_reduced_comparing-images_50.txt'

size_training_data = 32000
size_testing_data  =  4000

path_patches = "C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/"

def main():
	train_set = []
	test_set = []

	carnd_vehicles_data, total_carnd_vehicles = get_paths_and_label(path_carnd_vehicles, '1')
	train_set.extend(carnd_vehicles_data)
	carnd_nonvehicles_data, total_carnd_nonvehicles = get_paths_and_label(path_carnd_nonvehicles, '0')
	train_set.extend(carnd_nonvehicles_data)

	left_training_vehicles = int(abs((size_training_data / 2) - total_carnd_vehicles))
	left_training_nonvehicles = int(abs((size_training_data / 2) - total_carnd_nonvehicles))

	miotcd_vehicles_data, t = get_paths_and_label(path_miotcd_vehicles, '1', fileext='.jpg')
	random.shuffle(miotcd_vehicles_data)
	for i in range(left_training_vehicles):
			train_set.append(miotcd_vehicles_data[i])

	miotcd_nonvehicles_data, t = get_paths_and_label(path_miotcd_nonvehicles, '0', fileext='.jpg')
	random.shuffle(miotcd_nonvehicles_data)
	for i in range(left_training_nonvehicles):
			train_set.append(miotcd_nonvehicles_data[i])

	random.shuffle(train_set)

	print(len(train_set))
	for i in range(10):
		print(random.choice(train_set))

	# test set
	with open(info_filename, "rb") as fp:  # Unpickling
			images_info_reduced = pickle.load(fp)

	random.shuffle(images_info_reduced)
	empty_count = 0
	ocuppied_count = 0
	subsets_info = {'test':[]}

	for image_info in images_info_reduced:
		state = image_info['state']
		if state == '0':
			if empty_count < int(size_testing_data / 2):
				empty_count += 1
			else:
				continue
		elif state == '1':
			if ocuppied_count < int(size_testing_data / 2):
				ocuppied_count += 1
			else:
				continue
		image_path = os.path.join(path_patches, image_info['filePath'])
		test_set.append({'path': image_path, 'y': image_info['state']})
		subsets_info['test'].append(image_info)

	test_file = ntpath.basename(info_filename).split('.')[0]
	subset_dir = 'train_CarND-{}v-{}nv_MIO-TCD-{}v-{}nv_test_{}-{}'.format(total_carnd_vehicles, total_carnd_nonvehicles, left_training_vehicles, left_training_nonvehicles, test_file, len(test_set))
	create_subset_directorie(subset_dir)

	save_subsets_info(subsets=subsets_info, subsets_dir=subset_dir, extra_info='Special subset of training form of:\n vehicles {} from CarND-Project5 and {} from MIO-TCD \n nonvehicles {} from CarND-Project5 and {} from MIO-TCD'.format(total_carnd_vehicles, left_training_vehicles, total_carnd_nonvehicles, left_training_nonvehicles))

	save_subset(subset=train_set, subsets_dir=subset_dir, type='train')
	save_subset(subset=test_set, subsets_dir=subset_dir, type='test')

	print(len(test_set))
	for i in range(10):
		print(random.choice(test_set))









if __name__ == "__main__":
		main()

