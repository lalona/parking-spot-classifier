from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.layers.core import Layer
from keras import backend as K


def create_googlenet(width, height, depth, classes):
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
				inputShape = (depth, height, width)

		print(inputShape)

		# creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)

		input = Input(shape=inputShape)

		# Typical convolution layer
		conv1_7x7_s2 = Conv2D(6, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input)

		conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

		pool1_helper = PoolHelper()(conv1_zero_pad)

		pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(
				pool1_helper)

		conv2_3x3_reduce = Conv2D(2, kernel_size=(1, 1), padding='same', activation='relu', name='conv2/3x3_reduce')(
				pool1_3x3_s2)

		conv2_3x3 = Conv2D(6, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2/3x3')(
				conv2_3x3_reduce)

		conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_3x3)

		pool2_helper = PoolHelper()(conv2_zero_pad)

		pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2/3x3_s2')(
				pool2_helper)

		# Inception module
		inception_3a_1x1 = Conv2D(4, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3a/1x1')(pool2_3x3_s2)

		inception_3a_3x3_reduce = Conv2D(4, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3a/3x3_reduce')(pool2_3x3_s2)

		inception_3a_3x3 = Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', name='inception_3a/3x3')(inception_3a_3x3_reduce)

		inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool')(pool2_3x3_s2)

		inception_3a_pool_proj = Conv2D(6, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3a/pool_proj')(inception_3a_pool)

		inception_3a_output = concatenate([inception_3a_1x1, inception_3a_3x3, inception_3a_pool_proj], axis=3)

		pool5_7x7_s1 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3), name='pool5/7x7_s2')(inception_3a_output)

		loss3_flat = Flatten()(pool5_7x7_s1)

		pool5_drop_7x7_s1 = Dropout(0.2)(loss3_flat)

		loss3_classifier = Dense(classes, name='loss3/classifier')(pool5_drop_7x7_s1)

		loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

		googlenet = Model(input=input, output=loss3_classifier_act)

		return googlenet, 'googlenet'

class PoolHelper(Layer):

		def __init__(self, **kwargs):
				super(PoolHelper, self).__init__(**kwargs)

		def call(self, x, mask=None):
				return x[:, 1:, 1:]

		def get_config(self):
				config = {}
				base_config = super(PoolHelper, self).get_config()
				return dict(list(base_config.items()) + list(config.items()))