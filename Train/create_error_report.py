"""
El objetivo es que acomode en un archivo de texto los errores de menor a mayor
Que acomode los mejores en cnrpark, pklot y en diferencia entre los dos
"""
import os
import ntpath
import json
from tqdm import tqdm
import pickle
from operator import itemgetter
import argparse

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
		"""
		ap = argparse.ArgumentParser()
		ap.add_argument("-s", "--sorted-by", required=True,
										help="Sorted by")

		args = vars(ap.parse_args())

		dataset = args['sorted_by']
		"""
		ap = argparse.ArgumentParser()
		ap.add_argument("-e", "--experiment-dir", required=True,
										help="Path to the experiment dir")
		args = vars(ap.parse_args())
		experiment_dir = args['experiment-dir']

		error_images_files = []

		for (dirpath, dirnames, filenames) in os.walk(experiment_dir):
				for filename in filenames:
						if filename.startswith('error_images'):
								error_images_files.append(os.sep.join([dirpath, filename]))

		error_images_files_by_dataset = {}
		for error_image_file in error_images_files:
				dataset = ntpath.basename(error_image_file)
				if dataset not in error_images_files_by_dataset:
						error_images_files_by_dataset[dataset] = []
				error_images_files_by_dataset[dataset].append(error_image_file)

		error_images_files_by_dataset_error = {}

		for dataset, error_images_files_ in error_images_files_by_dataset.items():
				error_images_files_by_dataset_error[dataset] = []
				with open(getErrorFile(error_images_files_[0]), 'r') as f:
						try:
								error_images_info = json.load(f)
						except:
								print('fallo en: {}'.format(error_images_file))
				error_images_info['dataset']

				for error_images_file in tqdm(error_images_files_):

						with open(getErrorFile(error_images_file), 'r') as f:
								try:
									error_images_info = json.load(f)
								except:
									print('fallo en: {}'.format(error_images_file))
						datas = ''.join(error_images_file.split('\\')[8].split('_')[:-1])
						net = error_images_file.split('\\')[8].split('_')[-1]
						error_images_files_by_dataset_error[dataset].append({'error_images_file': error_images_file, 'error': error_images_info['error'], 'net': net, 'dataset': datas})
				error_images_files_by_dataset_error[dataset] = sorted(error_images_files_by_dataset_error[dataset], key=itemgetter('dataset', 'error'))

		net = ''
		with open('error_images_sorted2.txt', 'w+') as f:
				for dataset, error_images_files_sorted in error_images_files_by_dataset_error.items():
						f.write('Dataset tested: {} \n'.format(dataset))
						for error_images_file in error_images_files_sorted:
								if error_images_file['net'] != net:
										net = error_images_file['net']
										#f.write('Net: {} \n'.format(net))
								f.write('{} \n'.format(error_images_file['error_images_file']))
								f.write('{} \n'.format(error_images_file['error']))




if __name__ == "__main__":
		main()

