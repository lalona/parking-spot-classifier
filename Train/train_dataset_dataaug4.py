"""
This should receive the a file with the paths of the images for training his state an images for the validation
"""
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
from all_models import getModel
from datetime import datetime
import pandas as pd
from keras.callbacks import CSVLogger, Callback
import keras.backend as K
import csv

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_CHANNEL = 3
NUM_CLASSES = 2
EPOCHS = 2
INIT_LR = 1e-3
BATCH_SIZE = 256

learning_rates = []
class PrintLR(Callback):
		def on_epoch_end(self, epoch, logs=None):
				learning_rates.append(K.eval(self.model.optimizer.lr))
				print(K.eval(self.model.optimizer.lr))

def main():
		ap = argparse.ArgumentParser()
		ap.add_argument("-d", "--dataset", required=True,
										help="Path to input dataset directory")
		ap.add_argument("-n", "--net-model", required=True,
										help="Name of the net model.")
		ap.add_argument("-t", "--test-folder", required=True,
										help="Foldet to the test folder.")

		args = vars(ap.parse_args())

		dataset_directory = args['dataset']
		net_model = args['net_model']
		test_folder = args['test_folder']
		test_folder = os.path.join('test_folder', test_folder)
		if not os.path.isdir(test_folder):
			os.mkdir(test_folder)
		train_results_direcotry = os.path.join(test_folder,
																					 "{}_{}".format(os.path.basename(os.path.normpath(dataset_directory)),
																													net_model))
		if not os.path.isdir(train_results_direcotry):
			os.mkdir(train_results_direcotry)
		print(train_results_direcotry)

		# names of the history files
		model_path = os.path.join(train_results_direcotry, 'parking_classification.model')
		plot_path = os.path.join(train_results_direcotry, 'plot.png')
		train_details_path = os.path.join(train_results_direcotry, 'details.json')
		csv_log_path = os.path.join(train_results_direcotry, 'epochs_log.csv')
		csv_lr_path = os.path.join(train_results_direcotry, 'learning_rates.csv')

		# initialize the model
		print("[INFO] compiling model...")
		create_net = getModel(net_model)
		model, architecture_name = create_net(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, depth=IMAGE_CHANNEL, classes=NUM_CLASSES)

		opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.05)

		model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

		image_gen = ImageDataGenerator(rescale=1./255)
		image_gen_val = ImageDataGenerator(rescale=1. / 255)

		df_train = pd.read_csv(os.path.join(dataset_directory, 'data_paths_train.csv'))
		df_test = pd.read_csv(os.path.join(dataset_directory, 'data_paths_test.csv'))
		train_generator = image_gen.flow_from_dataframe(dataframe=df_train, x_col="path", y_col="y", directory=None,
																									class_mode="categorical", target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE)
		test_generator = image_gen_val.flow_from_dataframe(dataframe=df_test, x_col="path", y_col="y",
																										class_mode="categorical", directory=None, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
																										batch_size=BATCH_SIZE)

		# train the network
		print("[INFO] training network...")
		print(model.summary())
		printlr = PrintLR()
		callbacks = [CSVLogger(filename=csv_log_path, separator=',', append=False), printlr]#, history, lrate]
		STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
		STEP_SIZE_VALID = test_generator.n // test_generator.batch_size
		H = model.fit_generator(generator=train_generator,
														validation_data=test_generator,
														steps_per_epoch=STEP_SIZE_TRAIN,
														validation_steps=STEP_SIZE_VALID,
														epochs=EPOCHS,
														verbose=1,
														callbacks=callbacks)

		# save the model to disk
		print("[INFO] serializing network...")
		model.save(model_path)

		# plot the training loss and accuracy
		plt.style.use("ggplot")
		plt.figure()
		N = EPOCHS
		print(H.history)
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy on Ocuppied/Empty")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(plot_path)

		details = {
				'epochs': str(EPOCHS),
			 	'batch_size': str(BATCH_SIZE),
				'image_size': "{}x{}".format(IMAGE_WIDTH, IMAGE_HEIGHT),
				'arquitecture': architecture_name,
				'augmented_data': 'True'
		}
		with open(os.path.join(train_details_path), 'w') as outfile:
				json.dump(details, outfile)

		with open(csv_lr_path, 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(learning_rates)

if __name__ == "__main__":
		main()

