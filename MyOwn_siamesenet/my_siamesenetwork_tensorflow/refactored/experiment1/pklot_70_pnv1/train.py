from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import ParkinglotDataset
from model import mnist_model, contrastive_loss
import argparse
from tqdm import tqdm
import json
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 52, 'Batch size.')
flags.DEFINE_integer('train_iter', 1400, 'Total training iter')
flags.DEFINE_integer('step', 1, 'Save after ... iteration')
flags.DEFINE_string('dataset', None, 'Path to the directory containing dataset of training and tests')
flags.DEFINE_string('model', 'mnist', 'model to run')

loss_hisotry = []

if __name__ == "__main__":

		# setup dataset
		if FLAGS.model == 'mnist':
				dataset = ParkinglotDataset(FLAGS.dataset)
				model = mnist_model
				placeholder_shape = [None] + list((224, 224, 3))

				print("placeholder_shape", placeholder_shape)
				colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
									'#009999']
		else:
				raise NotImplementedError("Model for %s is not implemented yet" % FLAGS.model)

		# Setup network
		next_batch = dataset.get_siamese_batch
		left = tf.placeholder(tf.float32, placeholder_shape, name='left')
		right = tf.placeholder(tf.float32, placeholder_shape, name='right')
		with tf.name_scope("similarity"):
				label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
				label_float = tf.to_float(label)
		margin = 0.7
		left_output = model(left, reuse=False)
		right_output = model(right, reuse=True)
		#loss = tf.contrib.losses.metric_learning.contrastive_loss(label_float, left_output, right_output, margin=1.0)
		loss = contrastive_loss(left_output, right_output, label_float, margin)

		# Setup Optimizer
		global_step = tf.Variable(0, trainable=False)

		# starter_learning_rate = 0.0001
		# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
		# tf.scalar_summary('lr', learning_rate)
		# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)
		train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

		# Start Training
		saver = tf.train.Saver()
		info = {'dataset': FLAGS.dataset}
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())

				# setup tensorboard
				tf.summary.scalar('step', global_step)
				tf.summary.scalar('loss', loss)
				for var in tf.trainable_variables():
						tf.summary.histogram(var.op.name, var)
				merged = tf.summary.merge_all()
				writer = tf.summary.FileWriter('train.log', sess.graph)

				# train iter
				for i in tqdm(range(FLAGS.train_iter)):
						batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size)

						_, l, summary_str = sess.run([train_step, loss, merged],
																				 feed_dict={left: batch_left, right: batch_right, label: batch_similarity})

						writer.add_summary(summary_str, i)
						loss_hisotry.append(str(l))
						print("\r#%d - Loss" % i, l)

						"""
						if (i + 1) % FLAGS.step == 0:
							#generate test
							# TODO: create a test file and run per batch
							feat = sess.run(left_output, feed_dict={left:dataset.images_test})
			
							labels = dataset.labels_test
							# plot result
							f = plt.figure(figsize=(16,9))
							f.set_tight_layout(True)
							for j in range(10):
									plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(), '.', c=colors[j], alpha=0.8)
							plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
							plt.savefig('img/%d.jpg' % (i + 1))
						"""

				saver.save(sess, "model/model.ckpt")
		info['loss_history'] = loss_hisotry
		with open('info.json', 'w') as outfile:
				json.dump(info, outfile)




