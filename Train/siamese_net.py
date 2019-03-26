# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Dropout

class siamesenet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
		"""
		n.conv1 = L.Convolution(bottom="data", num_output=16, kernel_size=11, stride=4, **conv_defaults)
		n.relu1 = L.ReLU(n.conv1, in_place=True)
		n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.conv2 = L.Convolution(n.pool1, num_output=20, kernel_size=5, stride=1, **conv_defaults)
		n.relu2 = L.ReLU(n.conv2, in_place=True)
		n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.conv3 = L.Convolution(n.pool2, num_output=30, kernel_size=3, stride=1, **conv_defaults)
		n.relu3 = L.ReLU(n.conv3, in_place=True)
		n.pool3 = L.Pooling(n.relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.fc4 = L.InnerProduct(n.pool3, num_output=48, **fc_defaults)
		n.relu4 = L.ReLU(n.fc4, in_place=True)

		n.fc5 = L.InnerProduct(n.relu4, num_output=self.params['num_output'], **fc_defaults)
		"""
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(8, kernel_size=(7, 7), strides=(1, 1), padding="same", input_shape=inputShape, kernel_initializer='glorot_normal'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding="same", input_shape=inputShape,
										 kernel_initializer='glorot_normal'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_normal'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# third set of CONV => RELU => POOL layers
		model.add(Conv2D(64, (1, 1), strides=(1, 1), padding="same", kernel_initializer='glorot_normal'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(4, (1, 1), strides=(1, 1), padding="same", kernel_initializer='glorot_normal'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(48))
		model.add(Activation("relu"))
		#model.add(Dropout(0.2))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return [model, 'siamesenet']