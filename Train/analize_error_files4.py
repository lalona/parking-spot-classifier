"""
Si el conjunto que se uso para entrenar tiene la misma base que el conjunto que se uso para probar
se van a separar el error del subconjunto de entrenamiento y el error del subconjunto de validacion.
"""
import os
import ntpath
import json
from tqdm import tqdm
import pickle
from operator import itemgetter
import argparse
from astropy.table import Table, Column

def getErrorFile(err_file):
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file
		return err_file


def createErrorFile(err_file, failed_images):
		if len(err_file) >= 259:
				err_file = "\\\\?\\" + err_file

		with open(err_file, 'w') as outfile:
				json.dump(failed_images, outfile)

def main():
		error_images_files = []
		path = os.getcwd()

		experiments_errors = []
		for (dirpath, dirnames, filenames) in os.walk(path):
				for filename in filenames:
						if filename.startswith('error_images_info_'):
								experiment = dirpath.split('\\')[-2]
								error_images_file = os.sep.join([dirpath, filename])
								with open(getErrorFile(error_images_file), 'r') as f:
										try:
												error_images_info = json.load(f)
										except:
												print('fallo en: {}'.format(error_images_file))

								experiment_net = experiment.split('_')[-1]
								experiment_dataset = experiment.replace('_{}'.format(experiment_net), '')
								dataset = ntpath.basename(error_images_file)
								if 'pklot_labels_reduced_comparing-images_70' in dataset:
										dataset_tested = 'pklot_labels_reduced_comparing-images_70'
								elif 'cnrpark_labels_reduced_comparing-images_70' in dataset:
										dataset_tested = 'cnrpark_labels_reduced_comparing-images_70'

								if 'epoch' in dataset:
										epoch = dataset[dataset.index('epoch'):len('epoch0')]
								else:
										epoch = 'na'
								experiments_errors.append({'tested_on': dataset_tested, 'net': experiment_net, 'train_on': experiment_dataset, 'epoch': epoch, 'error': error_images_info['error']})
								experiments_errors = sorted(experiments_errors, key=itemgetter('net'))
								print(experiments_errors)




if __name__ == "__main__":
		main()

