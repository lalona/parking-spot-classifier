import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
import json
import os

IMAGE_HEIGHT = 0
IMAGE_WIDTH = 0
IMAGE_CHANNEL = 3
NUM_CLASSES = 2
ITERATIONS = 0
INIT_LR = 0
BATCH_SIZE = 0
MARGIN = 0.7
loss_hisotry = []

def experiment(dataset_dir, neural_net, train_results_direcotry):
		from constrastive_loss import contrastive_loss
		from siamesenet_dataset import ParkinglotDataset
		from all_psiamesenets import getModel
		import tensorflow as tf
		# setup dataset
		model_dir = os.path.join(train_results_direcotry, 'model')
		model_path = os.path.join(model_dir, 'model.ckpt')
		train_log = os.path.join(train_results_direcotry, 'train.log')
		info_json_path = os.path.join(train_results_direcotry, 'info.json')
		if os.path.isdir(model_dir):
				print('The experiment {} was already made'.format(train_results_direcotry))
				return False

		if not os.path.isdir(train_results_direcotry):
				os.mkdir(train_results_direcotry)
		print(train_results_direcotry)

		placeholder_shape = [None] + list((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
		dataset = ParkinglotDataset(dataset_dir, IMAGE_HEIGHT)
		model = getModel(neural_net)

		# Setup network
		next_batch = dataset.get_siamese_batch
		left = tf.placeholder(tf.float32, placeholder_shape, name='left')
		right = tf.placeholder(tf.float32, placeholder_shape, name='right')
		with tf.name_scope("similarity"):
				label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
				label_float = tf.to_float(label)


		left_output = model(left, reuse=False)
		right_output = model(right, reuse=True)

		loss = contrastive_loss(left_output, right_output, label_float, MARGIN)

		# Setup Optimizer
		global_step = tf.Variable(0, trainable=False)
		train_step = tf.train.AdamOptimizer(INIT_LR).minimize(loss, global_step=global_step)

		# Start Training
		saver = tf.train.Saver()
		info = {'dataset': dataset_dir, 'net': neural_net}
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())

				# setup tensorboard
				tf.summary.scalar('step', global_step)
				tf.summary.scalar('loss', loss)
				for var in tf.trainable_variables():
						tf.summary.histogram(var.op.name, var)
				merged = tf.summary.merge_all()
				writer = tf.summary.FileWriter(train_log, sess.graph)
				# train iter
				for i in tqdm(range(ITERATIONS)):
						batch_left, batch_right, batch_similarity = next_batch(BATCH_SIZE)

						_, l, summary_str = sess.run([train_step, loss, merged],
																				 feed_dict={left: batch_left, right: batch_right, label: batch_similarity})

						writer.add_summary(summary_str, i)
						loss_hisotry.append(str(l))
						print("\r#%d - Loss" % i, l)
				saver.save(sess, model_path)
		info['loss_history'] = loss_hisotry

		with open(info_json_path, 'w') as outfile:
				json.dump(info, outfile)

		tf.reset_default_graph()


def main():
		global ITERATIONS, INIT_LR, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, MARGIN
		ap = argparse.ArgumentParser()
		ap.add_argument("-e", "--experiments-specs", required=True,
										help="Path to the json file with the specification of the experiments")
		ap.add_argument("-d", "--datasets-dir", required=True,
										help="Path to the datasets directory")
		args = vars(ap.parse_args())
		experiments_specs_path = args['experiments_specs']
		datasets_dir = args['datasets_dir']

		with open(os.path.join(experiments_specs_path), 'r') as outfile:
				experiments_specs = json.load(outfile)
		training_params = experiments_specs['training_params']
		ITERATIONS = training_params['iter']
		INIT_LR = training_params['lr']
		IMAGE_HEIGHT = training_params['dim']
		IMAGE_WIDTH = training_params['dim']
		BATCH_SIZE = training_params['batch_size']
		MARGIN = training_params['margin']

		neural_nets = experiments_specs['neural_nets']
		neural_nets = neural_nets.split(',')

		datasets = experiments_specs['datasets']
		datasets = datasets.split(',')

		experiment_name = experiments_specs['experiment_name']

		test_folder = os.path.join('test_folder', experiment_name)
		experiment_specs_training_params_path = os.path.join(test_folder, 'training_params.json')
		if os.path.isfile(experiment_specs_training_params_path):
				with open(os.path.join(experiment_specs_training_params_path), 'r') as outfile:
						old_training_params = json.load(outfile)
						for key, param in old_training_params.items():
								if param != training_params[key]:
										print(
												'The old params differ from the one specified in the experiment specs file, you cant change this params for one experiment.')
										return
		else:
				with open(os.path.join(experiment_specs_training_params_path), 'w') as outfile:
						json.dump(training_params, outfile)

		if not os.path.isdir(test_folder):
				os.mkdir(test_folder)
		made_one = False
		for dataset in datasets:
				dataset_dir = os.path.join(datasets_dir, dataset)
				print('Making the training on the dataset: {}'.format(dataset_dir))
				for neural_net in neural_nets:
						print('Making the training with: {}'.format(neural_net))
						train_results_direcotry = os.path.join(test_folder,
																									 "{}_{}".format(os.path.basename(os.path.normpath(dataset_dir)),
																																	neural_net))
						model_dir = os.path.join(train_results_direcotry, 'model')
						if os.path.isdir(model_dir):
								print('The experiment {} was already made'.format(train_results_direcotry))
								continue
						else:
								made_one = True
						experiment(dataset_dir, neural_net, train_results_direcotry)
						break
				if made_one:
					break

if __name__ == "__main__":
		main()



