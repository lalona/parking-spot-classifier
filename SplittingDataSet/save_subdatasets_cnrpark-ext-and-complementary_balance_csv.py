"""
Voy a usar la base de datos cnrpark-ext pero voy a complementar con alguna base de datos
como CarND-Project5 o MIO-TCD

Si el tamaño de entrenamiento es 3000 y cnrpark cuenta con 15000 se va a rellenar con las otras bases de datos
de una forma balanceada
El conjunto de prueba se queda puramente con el de cnrpark
"""

import argparse
from save_subdatasets_utils import getSubsets, count_quantity_of_classes, get_paths_and_label, create_subset_directorie, saveSubsetsInfo, save_subset, save_subset
import random
import ntpath


total_size_training_data = 35000
#total_size_testing_data  =  4000

# this is in porcentage
size_training_data_cnrpark = 70
size_test_data_cnrpark = 30

path_carnd_vehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\CarND-Project5\\vehicles'
path_carnd_nonvehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\CarND-Project5\\non-vehicles'

path_miotcd_vehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\MIO-TCD\\train\\car'
path_miotcd_nonvehicles = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\MIO-TCD\\train\\background'


def main():
	parser = argparse.ArgumentParser(description='Select the type of reduced.')
	parser.add_argument("-f", "--filename", type=str, required=True,
												help='Path to the file the contains the dictionary with the info of the dataset reduced.')
	parser.add_argument("-ext", "--extra-dataset", type=str, required=True,
											help='Complementary dataset choosen for  balance it can be carnd or mio-tcd')
	args = vars(parser.parse_args())


	cnrpark_info_filename = args["filename"]
	complementary_dataset = args["extra_dataset"]

	cnrpark_data_paths, cnrpark_subsets_info = getSubsets(cnrpark_info_filename, size_training_data_cnrpark, size_test_data_cnrpark, 'cnrpark')
	train_set = cnrpark_data_paths['train']
	test_set = cnrpark_data_paths['test']
	cnrpark_empty_count, cnrpark_occupied_count = count_quantity_of_classes(cnrpark_data_paths['train'])

	if complementary_dataset == 'carnd':
			complementary_vehicles_path = path_carnd_vehicles
			complementary_nonvehicles_path = path_carnd_nonvehicles
	else:
			complementary_vehicles_path = path_miotcd_vehicles
			complementary_nonvehicles_path = path_miotcd_nonvehicles

	if complementary_dataset == 'carnd':
			ext = 'png'
	else:
			ext = 'jpg'
	complementary_vehicles_data, total_complementary_vehicles = get_paths_and_label(complementary_vehicles_path, '1', ext)
	complementary_nonvehicles_data, total_complementary_nonvehicles = get_paths_and_label(complementary_nonvehicles_path, '0', ext)

	total_size_training_data_empty = int(total_size_training_data / 2)
	total_size_training_data_occupied = int(total_size_training_data / 2)

	missing_training_data_empty = total_size_training_data_empty - cnrpark_empty_count
	missing_training_data_occupied = total_size_training_data_occupied - cnrpark_occupied_count

	random.shuffle(complementary_nonvehicles_data)
	for i in range(missing_training_data_empty):
			train_set.append(complementary_nonvehicles_data[i])
	random.shuffle(complementary_vehicles_data)
	for i in range(missing_training_data_occupied):
			train_set.append(complementary_vehicles_data[i])

	random.shuffle(train_set)
	random.shuffle(test_set)

	cnrpark_filename = ntpath.basename(cnrpark_info_filename).split('.')[0]
	subset_dir = cnrpark_filename + '-{}v-{}nv_complementary-{}-{}v-{}nv'.format(cnrpark_occupied_count, cnrpark_empty_count, complementary_dataset, missing_training_data_occupied, missing_training_data_empty)
	create_subset_directorie(subset_dir)

	saveSubsetsInfo(subsets=cnrpark_subsets_info, subsets_dir=subset_dir, database='cnrpark',
										extra_info='Training subset complemented with {} vehicles {} and nonvehicles {}'.format(complementary_dataset, missing_training_data_occupied, missing_training_data_empty))

	save_subset(subset=train_set, subsets_dir=subset_dir, type='train')
	save_subset(subset=test_set, subsets_dir=subset_dir, type='test')

if __name__ == "__main__":
		main()
