"""
Voy a usar la base de datos cnrpark-ext pero voy a complementar con alguna base de datos
como CarND-Project5 o MIO-TCD

Si el tamaño de entrenamiento es 3000 y cnrpark cuenta con 15000 se va a rellenar con las otras bases de datos
de una forma balanceada
El conjunto de prueba se queda puramente con el de cnrpark
"""

import argparse
from save_subdatasets_utils import getSubsets, count_quantity_of_classes, get_paths_and_label, create_subset_directorie, saveSubsetsInfo, save_subset, save_subset, getPathsAndLabelDataaug
import random
import ntpath
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm

total_size_training_data = 32000
#total_size_testing_data  =  5000

# this is in porcentage
init_por_training_data = 70
init_por_test_data = 100 - init_por_training_data

def main():
	parser = argparse.ArgumentParser(description='Select the type of reduced.')
	parser.add_argument("-f", "--filename", type=str, required=True,
												help='Path to the file the contains the dictionary with the info of the dataset reduced.')
	parser.add_argument("-d", "--database", type=str, required=True,
											help='Database it can be cnrpark or pklot')
	args = vars(parser.parse_args())
	info_filename = args["filename"]
	database = args["database"]

	data_paths, subsets_info = getSubsets(info_filename, init_por_training_data, init_por_test_data, database)
	train_set = data_paths['train']
	test_set = data_paths['test']
	empty_count, occupied_count = count_quantity_of_classes(data_paths['train'])

	random.shuffle(train_set)
	random.shuffle(test_set)

	total_size_training_data_empty = int(total_size_training_data / 2)
	total_size_training_data_occupied = int(total_size_training_data / 2)
	missing_training_data_empty = total_size_training_data_empty - empty_count
	missing_training_data_occupied = total_size_training_data_occupied - occupied_count

	filename = ntpath.basename(info_filename).split('.')[0]
	subset_dir = filename + '-{}v-{}nv_short'.format(total_size_training_data_occupied, total_size_training_data_empty)
	create_subset_directorie(subset_dir)

	print('{} {}'.format(missing_training_data_occupied, missing_training_data_empty))

	total_data_occupied = 0
	total_data_empty = 0

	train_set_used = []
	train_set_not_used = []
	empty_data_removed = 0
	occupied_data_removed = 0
	for train_data in tqdm(train_set):
			y = train_data['y']
			if y == '1':
					if total_data_occupied >= int(total_size_training_data / 2):
							train_set_not_used.append(train_data)
							occupied_data_removed += 1
							continue
					total_data_occupied += 1
			else:
					if total_data_empty >= int(total_size_training_data / 2):
							train_set_not_used.append(train_data)
							empty_data_removed += 1
							continue
					total_data_empty += 1
			train_set_used.append(train_data)

	print(train_set_used[0])
	print(train_set_not_used[0])

	random.shuffle(train_set_used)

	saveSubsetsInfo(subsets=subsets_info, subsets_dir=subset_dir, database=database,
									extra_info='Training subset short, this info is not accurate because the training set was modified \n The size of the training set is: {} The removed data is {} empty  {} occupied'.format(
											total_size_training_data, empty_data_removed, occupied_data_removed))


	save_subset(subset=train_set_used, subsets_dir=subset_dir, type='train')
	save_subset(subset=train_set_not_used, subsets_dir=subset_dir, type='train_notused')
	save_subset(subset=test_set, subsets_dir=subset_dir, type='test')


if __name__ == "__main__":
		main()
