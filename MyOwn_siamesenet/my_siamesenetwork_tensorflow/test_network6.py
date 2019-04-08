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


def getSubsets_info2(dataset_unprocess):
		dataset_info = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						x1_e = spaces_comparisions['comparing_with_empty']
						x1_o = spaces_comparisions['comparing_with_occupied']
						for data in spaces_comparisions['comparissions']:
								data_ = {}
								data_['X1_e'] = (x1_e)
								data_['X1_o'] = (x1_o)
								data_['X2'] = (data['comparing_to'])
								data_['Y'] = (int(data['state']))
								data_['data'] = (data)
								dataset_info.append(data_)
		return dataset_info

def createComparisionFile(dir, name_labels, comparision_data, comp_method):
		comparision_file = os.path.join(dir, 'comparision_{}_coccupied_images_info_{}.txt'.format(comp_method, name_labels))
		if len(comparision_file) >= 259:
				err_file = "\\\\?\\" + comparision_file
		else:
				err_file = comparision_file
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

		dataset_info = getSubsets_info2(dataset)

		img_placeholder = tf.placeholder(tf.float32, [None, 200, 200, 3], name='img')
		net = mnist_model(img_placeholder, reuse=False)
		saver = tf.train.Saver()
		failed = 0
		count = 0
		comp_method1 = 'cosine'
		comp_method2 = 'euclidean'
		comparision_values_euclidean = {}
		comparision_values_cosine = {}
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				ckpt = tf.train.get_checkpoint_state("model")
				saver.restore(sess, "model/model.ckpt")
				img_file_e = ''
				img_file_o = ''
				for data_info  in tqdm(dataset_info):
					idxs_left_empty, idxs_left_occupied, idxs_right, labels = [], [], [], []
					if img_file_e != data_info['X1_e']:
						img_file_e = data_info['X1_e']
						img_file_o = data_info['X1_o']
						token = '{},{}'.format(img_file_e, img_file_o)
						if token not in comparision_values_cosine:
								comparision_values_cosine[token] = []
						if token not in comparision_values_euclidean:
								comparision_values_euclidean[token] = []
						l_empty = getImage(img_file_e)
						idxs_left_empty.append(l_empty)
						left_empty = np.array(idxs_left_empty) / 255.0
						left_feat_empty = sess.run(net, feed_dict={img_placeholder: left_empty})

						l_occupied = getImage(img_file_o)
						idxs_left_occupied.append(l_occupied)
						left_empty = np.array(idxs_left_occupied) / 255.0
						left_feat_occupied = sess.run(net, feed_dict={img_placeholder: left_empty})

					img_file_compare = data_info['X2']
					y = data_info['Y']
					r = getImage(img_file_compare)


					idxs_right.append(r)
					labels.append(int(y))

					right = np.array(idxs_right) / 255.0

					right_feat = sess.run(net, feed_dict={img_placeholder: right})
					dist_empty = cdist(left_feat_empty, right_feat, comp_method1)
					dist_occupied = cdist(left_feat_occupied, right_feat, comp_method1)
					comparision_values_cosine[token].append(
							{'comparing_to': img_file_compare, 'similarity': int(y), 'pred_similarity_empty': dist_empty.ravel()[0], 'pred_similarity_occupied': dist_occupied.ravel()[0]})

					dist_empty = cdist(left_feat_empty, right_feat, comp_method2)
					dist_occupied = cdist(left_feat_occupied, right_feat, comp_method2)
					comparision_values_euclidean[token].append(
							{'comparing_to': img_file_compare, 'similarity': int(y), 'pred_similarity_empty': dist_empty.ravel()[0],
							 'pred_similarity_occupied': dist_occupied.ravel()[0]})

					if dist_empty.ravel()[0] <= dist_occupied.ravel()[0] and int(y) == 1:
							failed += 1
					elif dist_empty.ravel()[0] >= dist_occupied.ravel()[0] and int(y) == 0:
							failed += 1
					count += 1
					if count % 1000 == 0:
							print('error: {} {} {}'.format(failed * 100 / count, failed, count))


		name_labels = ntpath.basename(dataset_path).split('.')[0]
		createComparisionFile('', name_labels, comparision_values_cosine, comp_method1)
		createComparisionFile('', name_labels, comparision_values_euclidean, comp_method2)

if __name__ == "__main__":
		main()