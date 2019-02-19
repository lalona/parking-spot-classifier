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

total_size_training_data = 40000
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
	subset_dir = filename + '-{}v-{}_dataaug-{}v-{}nv'.format(occupied_count, empty_count, missing_training_data_occupied, missing_training_data_empty)
	create_subset_directorie(subset_dir)

	print('{} {}'.format(missing_training_data_occupied, missing_training_data_empty))
	datagen = ImageDataGenerator(width_shift_range=0.1, horizontal_flip=True, rotation_range=90)

	dataug_dir = os.path.join(subset_dir, 'dataaug')
	dataug_dir, maded = create_subset_directorie(dataug_dir)

	total_dataaug_occupied = 0
	total_dataaug_empty = 0
	for train_data in tqdm(train_set):
			img_path = train_data['path']
			y = train_data['y']
			if y == '0':
					if missing_training_data_empty <= 0:
							continue
					missing_training_data_empty -= 1
					total_dataaug_empty += 1
			else:
					if missing_training_data_occupied <= 0:
							continue
					missing_training_data_occupied -= 1
					total_dataaug_occupied += 1
			img = load_img(img_path)  # this is a PIL image
			x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
			x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

			for batch in datagen.flow(x, batch_size=1,
																save_to_dir=dataug_dir, save_prefix=y, save_format='jpeg'):
					dataaug_subset, total_dataaug = getPathsAndLabelDataaug(dataug_dir, fileext='.jpeg')
					if total_dataaug < (total_dataaug_occupied + total_dataaug_empty):
							print('{} {} {}'.format(total_dataaug, total_dataaug_occupied, total_dataaug_empty))
					else:
							break



	dataaug_subset, total_dataaug = getPathsAndLabelDataaug(dataug_dir, fileext='.jpeg')
	print(total_dataaug)
	train_set.extend(dataaug_subset)
	random.shuffle(train_set)

	saveSubsetsInfo(subsets=subsets_info, subsets_dir=subset_dir, database=database,
										extra_info='Training subset complemented with {} vehicles {} and nonvehicles {}'.format('data augmentation', total_dataaug_occupied, total_dataaug_empty))


	save_subset(subset=train_set, subsets_dir=subset_dir, type='train')
	save_subset(subset=test_set, subsets_dir=subset_dir, type='test')

if __name__ == "__main__":
		main()
