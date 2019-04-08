from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS


def psiamesenet_v4(input, reuse=False):
		with tf.name_scope("model"):
				with tf.variable_scope("conv1") as scope:
						net = tf.contrib.layers.conv2d(input, 4, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)
						net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

				with tf.variable_scope("conv2") as scope:
						net = tf.contrib.layers.conv2d(net, 16, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse, stride=2)
						net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

				with tf.variable_scope("conv3") as scope:
						net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse, stride=2)
						net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

				with tf.variable_scope("conv4_1") as scope:
						net = tf.contrib.layers.conv2d(net, 8, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)

				with tf.variable_scope("conv4") as scope:
						net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)
						net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')


				with tf.variable_scope("conv5_1") as scope:
						net = tf.contrib.layers.conv2d(net, 16, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)

				with tf.variable_scope("conv5") as scope:
						net = tf.contrib.layers.conv2d(net, 56, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)

				with tf.variable_scope("conv6") as scope:
						net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
																					 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
																					 scope=scope, reuse=reuse)

				net = tf.contrib.layers.flatten(net)

		return net



