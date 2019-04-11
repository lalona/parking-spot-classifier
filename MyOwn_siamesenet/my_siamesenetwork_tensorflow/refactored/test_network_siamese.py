import argparse
import os
import json
from tqdm import tqdm
from siamesenet_dataset import getImage
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import ntpath
from all_psiamesenets import getModel

def getSubsets_info2(dataset_unprocess):
		dataset_info = []
		for parkinglot, spaces in dataset_unprocess.items():
				for space, spaces_comparisions in tqdm(spaces.items()):
						x1_e = spaces_comparisions['comparing_with_empty']['filepath']
						x1_o = spaces_comparisions['comparing_with_occupied']['filepath']
						for data in spaces_comparisions['comparissions']:
								data_ = {}
								data_['X1_e'] = (x1_e)
								data_['X1_o'] = (x1_o)
								data_['X2'] = (data['filepath'])
								data_['Y'] = (int(data['state']))
								data_['data'] = (data)
								dataset_info.append(data_)
		return dataset_info


def createJSONFile(dir, name_labels, json_data, prefix):
		comparision_file = os.path.join(os.getcwd(), dir, '{}_{}.txt'.format(prefix, name_labels))
		if len(comparision_file) >= 259:
				err_file = "\\\\?\\" + comparision_file
		else:
				err_file = comparision_file
		with open(err_file, 'w') as outfile:
				json.dump(json_data, outfile)


def test(dataset_path, dim, experiment_path):
		model_path = os.path.join(experiment_path, 'model', 'model.ckpt')
		info_path = os.path.join(experiment_path, 'info.json')
		test_dir = os.path.join(experiment_path, 'test_info')
		if not os.path.isdir(test_dir):
				os.mkdir(test_dir)
		with open(dataset_path) as f:
				dataset = json.load(f)

		with open(info_path) as f:
				info = json.load(f)

		dataset_info = getSubsets_info2(dataset)

		img_placeholder = tf.placeholder(tf.float32, [None, dim, dim, 3], name='img')
		model = getModel(info['net'])
		net = model(img_placeholder, reuse=False)
		saver = tf.train.Saver()
		failed = 0
		count = 0
		comparision_values_euclidean = {}

		failed_images = {'dataset': dataset_path, 'error': 0, 'image_info': []}
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				saver.restore(sess, model_path)
				img_file_e = ''
				img_file_o = ''
				for data_info in tqdm(dataset_info):
						idxs_left_empty, idxs_left_occupied, idxs_right, labels = [], [], [], []
						if img_file_e != data_info['X1_e']:
								img_file_e = data_info['X1_e']
								img_file_o = data_info['X1_o']
								token = '{},{}'.format(img_file_e, img_file_o)
								if token not in comparision_values_euclidean:
										comparision_values_euclidean[token] = []
								l_empty = getImage(img_file_e, dim)
								idxs_left_empty.append(l_empty)
								left_empty = np.array(idxs_left_empty) / 255.0
								left_feat_empty = sess.run(net, feed_dict={img_placeholder: left_empty})

								l_occupied = getImage(img_file_o, dim)
								idxs_left_occupied.append(l_occupied)
								left_empty = np.array(idxs_left_occupied) / 255.0
								left_feat_occupied = sess.run(net, feed_dict={img_placeholder: left_empty})

						img_file_compare = data_info['X2']
						y = data_info['Y']
						r = getImage(img_file_compare, dim)

						idxs_right.append(r)
						labels.append(int(y))

						right = np.array(idxs_right) / 255.0

						right_feat = sess.run(net, feed_dict={img_placeholder: right})

						dist_empty = cdist(left_feat_empty, right_feat, 'euclidean')
						dist_occupied = cdist(left_feat_occupied, right_feat, 'euclidean')
						comparision_values_euclidean[token].append(
								{'comparing_to': img_file_compare, 'similarity': int(y), 'pred_similarity_empty': dist_empty.ravel()[0],
								 'pred_similarity_occupied': dist_occupied.ravel()[0]})

						if dist_empty.ravel()[0] <= dist_occupied.ravel()[0] and int(y) == 1:
								failed += 1
								failed_images['image_info'].append(data_info['data'])
						elif dist_empty.ravel()[0] >= dist_occupied.ravel()[0] and int(y) == 0:
								failed += 1
								failed_images['image_info'].append(data_info['data'])
						count += 1
						if count % 1000 == 0:
								print('error: {} {} {}'.format(failed * 100 / count, failed, count))
		failed_images['error'] = failed * 100 / count
		name_labels = ntpath.basename(dataset_path).split('.')[0]
		createJSONFile(test_dir, name_labels, comparision_values_euclidean, prefix='distances_{}'.format('euclidean'))
		createJSONFile(test_dir, name_labels, failed_images, prefix='error_info_')

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-d", "--dataset", type=str, required=True,
												help='Path to dataset.')
		parser.add_argument("-dim", "--dimesions-img", type=int, required=True,
												help='Dimensions of image.')
		parser.add_argument("-e", "--experiment", type=str, required=True,
												help='Experiment.')

		args = vars(parser.parse_args())

		dataset_path = args["dataset"]
		dim = args["dimesions_img"]
		experiemnt_path = args["experiment"]

		test(dataset_path, dim, experiemnt_path)

if __name__ == "__main__":
		main()