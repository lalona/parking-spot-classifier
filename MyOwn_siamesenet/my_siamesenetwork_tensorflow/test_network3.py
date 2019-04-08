import argparse
import os
import json
from tqdm import tqdm
from keras.models import load_model
from dataset import getImage
import tensorflow as tf
import numpy as np
from model import *
from scipy.spatial.distance import cdist
import ntpath

IMG_HEIGHT = 200
IMG_WIDTH = 200
def getSubsets_info(dataset_unprocess):

		dataset_info = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						for data in spaces_comparisions:
								data_ = {}
								data_['X1'] = (data['comparing_with'])
								data_['X2'] = (data['comparing_to'])
								data_['Y'] = (int(data['state']))
								data_['data'] = (data)
								dataset_info.append(data_)
		return dataset_info


def createComparisionFile(dir, name_labels, comparision_data):
		comparision_file = os.path.join(dir, 'comparision_images_info_{}.txt'.format(name_labels))
		if len(comparision_file) >= 259:
				err_file = "\\\\?\\" + comparision_file
		with open(err_file, 'w') as outfile:
				json.dump(comparision_data, outfile)


def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-d", "--dataset", type=str, required=True,
												help='Path to dataset.')

		args = vars(parser.parse_args())

		dataset_path = args["dataset"]

		with open(dataset_path) as f:
				dataset = json.load(f)

		dataset_info = getSubsets_info(dataset)

		img_placeholder = tf.placeholder(tf.float32, [None, 200, 200, 3], name='img')
		net = mnist_model(img_placeholder, reuse=False)
		saver = tf.train.Saver()
		failed = 0
		count = 0
		empty_total = 0
		occupied_total = 0
		empty_count = 0
		occupied_count = 0
		empty_min = 1.0
		occupied_max = 0.0
		empty_error = 0.0
		occupied_error = 0.0
		failed_empty_count = 0
		failed_occupied_count = 0

		comparision_values = {}
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				ckpt = tf.train.get_checkpoint_state("model")
				saver.restore(sess, "model/model.ckpt")
				img_file = ''
				for data_info  in tqdm(dataset_info):
					idxs_left, idxs_right, labels = [], [], []
					if img_file != data_info['X1']:
						print('X1 changed')
						img_file = data_info['X1']
						if img_file not in comparision_values:
								comparision_values[img_file] = []
						l = getImage(img_file)




					img_file_compare = data_info['X2']
					y = data_info['Y']
					r = getImage(img_file_compare)

					idxs_left.append(l)
					idxs_right.append(r)
					labels.append(int(y))
					left = np.array(idxs_left) / 255.0
					right = np.array(idxs_right) / 255.0

					left_feat = sess.run(net, feed_dict={img_placeholder: left})
					right_feat = sess.run(net, feed_dict={img_placeholder: right})
					dist = cdist(left_feat, right_feat, 'cosine')
					comparision_values[img_file].append(
							{'comparing_to': img_file_compare, 'similarity': int(y), 'pred_similarity': dist.ravel()[0]})
					if int(y) == 1:
							empty_total += dist.ravel()[0]
							empty_count += 1
							if dist.ravel()[0] < empty_min:
									empty_min = dist.ravel()[0]
					else:
							occupied_total += dist.ravel()[0]
							occupied_count += 1
							if dist.ravel()[0] > occupied_max:
									occupied_max = dist.ravel()[0]
					treshold = 0.04
					if dist.ravel()[0] < treshold and int(y) == 1:
							failed += 1
							#empty_error += dist.ravel()[0]
							empty_error += 1
							failed_empty_count += 1
					if dist.ravel()[0] >= treshold and int(y) == 0:
							failed += 1
							#occupied_error += dist.ravel()[0]
							failed_occupied_count += 1
							occupied_error += 1

					#print('\n {} {}'.format(y, dist.ravel()[0]))
					count += 1
					if count % 100 == 0:
							#print('failed: {}'.format(failed_empty_count))
							print('error: {} {} c {} e'.format(failed * 100 / count, occupied_error, empty_error))
							#print('empty_avr: {} min: {} err: {}'.format(empty_total / empty_count, empty_min, empty_error / failed_empty_count))
							print('empty_avr: {}'.format(empty_total / empty_count))
							#print('occupied_avr: {} max: {} err: {}'.format(occupied_total / occupied_count, occupied_max, occupied_error / failed_occupied_count))
							print('occupied_avr: {}'.format(occupied_total / occupied_count))
							empty_min = 1.0
							occupied_max = 0.0
							empty_error = 0.0
							occupied_error = 0.0
							empty_total = 0
							occupied_total = 0
							empty_count = 0
							occupied_count = 0
							failed_empty_count = 0
							failed_occupied_count = 0


		name_labels = ntpath.basename(dataset_path).split('.')[0]
		createComparisionFile('', name_labels, comparision_values)

if __name__ == "__main__":
		main()