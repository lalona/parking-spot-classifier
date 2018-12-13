# Save augmented images to file
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
from keras import backend as K
import pandas as pd
import argparse
import csv
from tqdm import tqdm
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
								help="path to input dataset")

args = vars(ap.parse_args())

dataset_path = args['dataset']

# configure batch size and retrieve one batch of images
augmented_images_dir = os.path.join(dataset_path, 'augmented_images')

if not os.path.isdir(augmented_images_dir):
	os.makedirs(os.path.join(dataset_path, 'augmented_images'))


image_gen = ImageDataGenerator(rotation_range=50, horizontal_flip=True)
traindf = pd.read_csv(os.path.join(dataset_path, 'data_paths_train.csv'))
train_generator = image_gen.flow_from_dataframe(dataframe=traindf, directory=None, x_col='path', y_col='y', batch_size=100, save_to_dir=augmented_images_dir, save_prefix='aug', save_format='jpg')
data_paths = []
print(os.path.join('C:\\Eduardo\\Level1\\DeepLearning_Keras\\my_code\\ProyectoFinal\\Train', dataset_path, 'data_paths_aug.csv'))
count = 0
for i in tqdm(train_generator):
		idx = (train_generator.batch_index - 1) * train_generator.batch_size
		filenames = train_generator.filenames[idx: idx + train_generator.batch_size]
		classes = train_generator.classes[idx: idx + train_generator.batch_size]
		for file, y in zip(filenames, classes):
			data_path_state = {'path': os.path.join(augmented_images_dir, file), 'y': int(y)}
			data_paths.append(data_path_state)
		count += 1
		if count > 10:
				break

keys = data_paths[0].keys()
with open(os.path.join('C:\\Eduardo\\Level1\\DeepLearning_Keras\\my_code\\ProyectoFinal\\Train', dataset_path, 'data_paths_aug.csv'), 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, lineterminator='\n', fieldnames=keys)
		dict_writer.writeheader()
		dict_writer.writerows(data_paths)


