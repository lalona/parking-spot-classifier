"""
Esta es la primer version basada en Xception
Utiliza dropout y BatchNormalization
Utiliza la ultima capa de mAlexNet
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, concatenate, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model

from keras.layers.core import Layer
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, add, SeparableConv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout

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
		x = Conv2D(8, (3, 3),
											strides=(2, 2),
											use_bias=False,
											name='block1_conv1')(img_input)
		x = BatchNormalization(name='block1_conv1_bn')(x)
		x = Activation('relu', name='block1_conv1_act')(x)
		x = Conv2D(16, (3, 3),
											strides=(4, 4),
											use_bias=False,
											name='block1_conv2')(x)
		x = BatchNormalization(name='block1_conv2_bn')(x)
		x = Activation('relu', name='block1_conv2_act')(x)

		# 1
		residual = Conv2D(8, (1, 1),
														 strides=(2, 2),
														 padding='same',
														 use_bias=False)(x)
		residual = BatchNormalization()(residual)

		x = SeparableConv2D(8, (3, 3),
															 padding='same',
															 use_bias=False,
															 name='block2_sepconv1')(x)
		x = BatchNormalization(name='block2_sepconv1_bn')(x)
		x = Activation('relu', name='block2_sepconv2_act')(x)
		x = SeparableConv2D(8, (3, 3),
															 padding='same',
															 use_bias=False,
															 name='block2_sepconv2')(x)
		x = BatchNormalization(name='block2_sepconv2_bn')(x)

		x = MaxPooling2D((3, 3),
														strides=(2, 2),
														padding='same',
														name='block2_pool')(x)
		x = add([x, residual])

		# third part
		residual = Conv2D(16, (1, 1), strides=(2, 2),
														 padding='same', use_bias=False)(x)
		residual = BatchNormalization()(residual)

		x = Activation('relu', name='block3_sepconv1_act')(x)
		x = SeparableConv2D(16, (3, 3),
															 padding='same',
															 use_bias=False,
															 name='block3_sepconv1')(x)
		x = BatchNormalization(name='block3_sepconv1_bn')(x)
		x = Activation('relu', name='block3_sepconv2_act')(x)
		x = SeparableConv2D(16, (3, 3),
															 padding='same',
															 use_bias=False,
															 name='block3_sepconv2')(x)
		x = BatchNormalization(name='block3_sepconv2_bn')(x)

		x = MaxPooling2D((3, 3), strides=(2, 2),
														padding='same',
														name='block3_pool')(x)
		x = add([x, residual])

		x = Flatten()(x)
		x = Dense(48, activation='relu')(x)
		x = Dropout(0.2)(x)
		# softmax classifier
		out = Dense(classes, activation='softmax')(x)

		model = Model(inputs=img_input, outputs=out)
		return model, 'parkingnetv4'