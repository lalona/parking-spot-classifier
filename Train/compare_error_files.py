import argparse
import json
import os

def main():
	print('hola')
	ap = argparse.ArgumentParser()
	ap.add_argument('-e1', '--error-file1', type=str, required=True, help='The path to the file that contains the error images')
	ap.add_argument('-e2', '--error-file2', type=str, required=True,
									help='The path to the file that contains the error images')

	args = vars(ap.parse_args())

	error_file1 = args['error_file1']
	error_file2 = args['error_file2']

	with open(error_file1, 'r') as fjson:
			failed_images1 = json.load(fjson)

	with open(error_file2, 'r') as fjson2:
			failed_images2 = json.load(fjson2)


	failed_images3 = {'comparing_files': "{} with {}".format(error_file1, error_file2), 'image_info': []}
	print(failed_images2['image_info'])
	for failed_image in failed_images1['image_info']:
			for failed_image2 in failed_images2['image_info']:
				if failed_image['filepath'] == failed_image2['filepath'] and failed_image['filename'] == failed_image2['filename']:
						failed_images3['image_info'].append(failed_image)
	print("Failed image 1: {} 2: {} 3: {}".format(len(failed_images1['image_info']), len(failed_images2['image_info']), len(failed_images3)))

	error_path1, error_filename1 = os.path.split(error_file2)
	with open(os.path.join(error_path1, 'error_file_comparision.txt'), 'w') as outfile:
			json.dump(failed_images3, outfile)

if __name__ == "__main__":
		main()