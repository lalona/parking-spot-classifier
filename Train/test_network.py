from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

import pickle
import os
from tqdm import tqdm
import json
import ntpath
from keras.layers.core import Layer

def main():
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--model", required=True,
										help="path to trained model model")
		#ap.add_argument("-i", "--image", required=False,
		#								help="path to input image")

		ap.add_argument("-inf", "--images-info", type=str, required=True,
												help='Path to the file the contains the dictionary with the info of the dataset reduced.')
		ap.add_argument("-dim", "--dimension", type=int, required=True,
										help='The size of width and height.')
		ap.add_argument("-d", "--dataset", type=str, required=True,
										help='It can be pklot or cnrpark.')



		args = vars(ap.parse_args())

		images_info_path = args['images_info']

		dataset = args['dataset']

		dim = args['dimension']

		if dataset != 'pklot' and dataset != 'cnrpark':
				print('The dataset can only be pklot o cnrpark')
				return

		with open(images_info_path, "rb") as fp:   # Unpickling
				images_info_reduced = pickle.load(fp)

		# load the trained convolutional neural network
		print("[INFO] loading network...")
		model_path = args["model"]
		model = load_model(model_path,custom_objects={'PoolHelper': PoolHelper})
		model_onlypath, model_filename = os.path.split(model_path)
		print(os.path.join(model_onlypath, 'error_images_info_pklot_labels.txt'))

		failed_images = {'dataset': images_info_path, 'image_info': []}
		count = 0
		good = 0
		error = 0
		total_count = 0
		for image_info in tqdm(images_info_reduced):

				# load the image
				if dataset == 'pklot':
					path_image = os.path.join('C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot', image_info["filepath"], image_info["filename"])
				else:
						path_image = os.path.join("C:/Eduardo/ProyectoFinal/Datasets/CNR-EXT/PATCHES/", image_info['filePath'])
				image = cv2.imread(path_image)
				orig = image.copy()

				# pre-process the image for classification
				image = cv2.resize(image, (dim, dim))
				image = image.astype("float") / 255.0
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)

				# classify the input image
				(empty, ocuppied) = model.predict(image)[0]

				# build the label
				label = "Occupied" if ocuppied > empty else "Empty"
				proba = ocuppied if ocuppied > empty else empty
				label = "{}: {:.2f}%".format(label, proba * 100)


				#if (int(image_info['state']) == 0) != (ocuppied > empty):
				if (int(image_info['state']) == 0) != (ocuppied < empty): # esta es la corrrecta

						# draw the label on the image
						"""
						output = imutils.resize(orig, width=400)
						cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
												0.7, (0, 255, 0), 2)

						# show the output image
						cv2.imshow("Output", output)
						cv2.waitKey(0)
						"""
						image_info['proba_empty'] = str(empty)
						image_info['proba_ocuppied'] = str(ocuppied)
						image_info['whole_path'] = path_image
						failed_images['image_info'].append(image_info)
						error += 1
				else:
						good += 1
				count += 1
				if count == 1000:
						por_error = (len(failed_images['image_info']) * 100) / total_count
						print("In a total of 1000 error: {} total of error images: {} error por.: {}".format(error, len(failed_images['image_info']), por_error))
						count = 0
						error = 0
						good = 0
				total_count += 1

		por_error = (len(failed_images['image_info']) * 100) / len(images_info_reduced)

		print("In a total of {} error: {} error por.: {}".format(len(images_info_reduced), len(
						failed_images['image_info']), por_error))

		name_labels = ntpath.basename(images_info_path).split('.')[0]
		test_dir = os.path.join(model_onlypath, 'test_info')
		if not os.path.isdir(test_dir):
			os.mkdir(test_dir)

		with open(os.path.join(test_dir, 'error_images_info_{}.txt'.format(name_labels)), 'w') as outfile:
				json.dump(failed_images, outfile)


class PoolHelper(Layer):

		def __init__(self, **kwargs):
				super(PoolHelper, self).__init__(**kwargs)

		def call(self, x, mask=None):
				return x[:, 1:, 1:]

		def get_config(self):
				config = {}
				base_config = super(PoolHelper, self).get_config()
				return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
		main()

