from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def contrastive_loss(model1, model2, y, margin):
		with tf.name_scope("contrastive-loss"):
				distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
				similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
				dissimilarity = (1 - y
												 ) * tf.square(tf.maximum((margin - distance),
																									0))  # give penalty to dissimilar label if the distance is bigger than margin
				return tf.reduce_mean(dissimilarity + similarity) / 2
