"""
Va a ser igual que parkingenv2 pero en la ultima capa no va a usar flatten va a usar global
y se agrega una capa extra antes del global
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, concatenate, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model

from keras.layers.core import Layer
from keras import backend as K

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def create_parkingnet(width, height, depth, classes):
		# creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)

		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
				inputShape = (depth, height, width)
				channel_axis = 1
		else:
				channel_axis = 3

		print(inputShape)

		# creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
		img_input = Input(shape=inputShape)
		# divide la original entre 4 excepto la primera porque el stride en lugar de ser 2x2 fue 4x4
		x = conv2d_bn(img_input, 16, 3, 3, strides=(4, 4), padding='valid')
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		branch1x1 = conv2d_bn(x, 4, 1, 1)

		branch5x5 = conv2d_bn(x, 3, 1, 1)
		branch5x5 = conv2d_bn(branch5x5, 4, 5, 5)

		branch3x3dbl = conv2d_bn(x, 4, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 6, 3, 3)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 6, 3, 3)

		branch_pool = AveragePooling2D((3, 3),
																					strides=(1, 1),
																					padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 2, 1, 1)
		x = concatenate(
				[branch1x1, branch5x5, branch3x3dbl, branch_pool],
				axis=channel_axis,
				name='mixed0')

		#
		branch3x3 = conv2d_bn(x, 24, 3, 3, strides=(4, 4), padding='valid')

		branch3x3dbl = conv2d_bn(x, 8, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 12, 3, 3)
		branch3x3dbl = conv2d_bn(
				branch3x3dbl, 12, 3, 3, strides=(4, 4), padding='valid')

		branch_pool = MaxPooling2D((3, 3), strides=(4, 4))(x)
		x = concatenate(
				[branch3x3, branch3x3dbl, branch_pool],
				axis=channel_axis,
				name='mixed3')

		#
		branch1x1 = conv2d_bn(x, 20, 1, 1)

		branch3x3 = conv2d_bn(x, 24, 1, 1)
		branch3x3_1 = conv2d_bn(branch3x3, 24, 1, 3)
		branch3x3_2 = conv2d_bn(branch3x3, 24, 3, 1)
		branch3x3 = concatenate(
				[branch3x3_1, branch3x3_2],
				axis=channel_axis,
				name='mixed9_1')
		branch_pool = AveragePooling2D(
				(3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 12, 1, 1)
		x = concatenate(
				[branch1x1, branch3x3, branch_pool],
				axis=channel_axis,
				name='mixed9')

		x = GlobalAveragePooling2D(name='avg_pool')(x)
		x = Dropout(0.2)(x)
		# softmax classifier
		out = Dense(classes, activation='softmax')(x)

		model = Model(inputs=img_input, outputs=out)
		return model, 'parkingnetv2'

