import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import MNISTDataset
from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec

dataset = MNISTDataset()
train_images = dataset.images_train[:20000]
test_images = dataset.images_test
len_test = len(train_images)
len_train = len(test_images)

#helper function to plot image
def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(data[idxs[i],:,:,0])
        ax.axis('off')
    plt.show()

img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='img')
net = mnist_model(img_placeholder, reuse=False)

saver = tf.train.Saver()
with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state("model")
		saver.restore(sess, "model/model.ckpt")

		train_feat = sess.run(net, feed_dict={img_placeholder: train_images[:10000]})

#generate new random test image
idx = np.random.randint(0, len_test)
im = test_images[idx]

#show the test image
show_image(idx, test_images)
print("This is image from id:", idx)

# run the test image through the network to get the test features
saver = tf.train.Saver()
with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state("model")
		saver.restore(sess, "model/model.ckpt")
		search_feat = sess.run(net, feed_dict={img_placeholder: [im]})

# calculate the cosine similarity and sort
dist = cdist(train_feat, search_feat, 'cosine')
rank = np.argsort(dist.ravel())

# show the top n similar image from train data
n = 7
show_image(rank[:n], train_images)
print("retrieved ids:", rank[:n])