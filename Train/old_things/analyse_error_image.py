import argparse
import json
from tqdm import tqdm
import os
import cv2
import imutils
def main():
	print('hola')
	ap = argparse.ArgumentParser()
	ap.add_argument('-e', '--error-file', type=str, required=True, help='The path to the file that contains the error images')
	ap.add_argument("-d", "--dataset", type=str, required=True, help='It can be pklot or cnrpark.')
	args = vars(ap.parse_args())

	error_file = args['error_file']
	dataset = args['dataset']

	with open(error_file, 'r') as fjson:
			failed_images = json.load(fjson)

	stadistics = {}

	for key, value in failed_images['image_info'][0].items():
		stadistics[key] = {}

	high_error_images = []
	for failed_image in tqdm(failed_images['image_info']):
			for key, value in failed_image.items():
					# Si ya se habia registro ese valor entonces se suma uno si no se le asigna cero
					if value in stadistics[key]:
							stadistics[key][value] += 1
					else:
							stadistics[key][value] = 1

					if key == 'proba_empty' or key == 'proba_ocuppied':
							if float(value) > 0.9:
									"""
									path_image = os.path.join('C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot', failed_image["filepath"],
																						failed_image["filename"])
									label = failed_image["state"]
									image = cv2.imread(path_image)
									#orig = cv2.imread(path_image)
									output = imutils.resize(image, width=400)
									cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
															0.7, (0, 255, 0), 2)
									# show the output image
									cv2.imshow("Output", output)
									cv2.waitKey(0)
									"""
									high_error_images.append(failed_image)

	for key, value in stadistics.items():
			#print(key)
			for k, v in value.items():
				if v > 200:
					print("From: {} attrib: {} count: {}".format(key, k, v))
	print('From total: {}'.format(len(failed_images['image_info'])))
	print('Total high error image: {}'.format(len(high_error_images)))

	for err_image in high_error_images:
			print(err_image)
			image_path = err_image['whole_path']
			image = cv2.imread(image_path)
			if dataset == 'cnrpark':
				cv2.imshow(err_image['filePath'], image)
			else:
				cv2.imshow(err_image['filepath'] + "" + err_image['filename'], image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

if __name__ == "__main__":
		main()