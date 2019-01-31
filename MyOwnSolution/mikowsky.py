import argparse
from compare_image_utils import get_images_info_by_parkinglot
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
from scipy.spatial import distance
import numpy as np

def try_opencv_method(empty_image, comparision_image, state):
	empty_image = cv2.cvtColor(empty_image, cv2.COLOR_BGR2HSV)
	comparision_image = cv2.cvtColor(comparision_image, cv2.COLOR_BGR2HSV)

	comparision_image[:, :, 2] = empty_image[:, :, 2]

	comparision_image = cv2.cvtColor(comparision_image, cv2.COLOR_HSV2BGR)
	empty_image = cv2.cvtColor(empty_image, cv2.COLOR_HSV2BGR)

	# initialize the results figure
	fig = plt.figure('Show image')
	fig.suptitle('sh', fontsize = 20)

	# loop over the results

	# show the result
	ax = fig.add_subplot(1, 2, 1)
	ax.set_title('empty image')
	plt.imshow(empty_image)
	plt.axis("off")

	ax = fig.add_subplot(1, 2, 2)
	ax.set_title('compar image {}'.format(state))
	plt.imshow(comparision_image)
	plt.axis("off")
	plt.show()

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-f", "--filename", type=str, required=True,
												help='Path to the file the contains the dictionary with the info of the dataset reduced.')

		args = vars(parser.parse_args())
		info_filename = args["filename"]

		images_info_by_patkinglot = get_images_info_by_parkinglot(info_filename)

		for parkinglot, images_info_by_spaces in images_info_by_patkinglot.items():
				for space, images_info_of_space in images_info_by_spaces.items():
						empty_space_filepath = ''
						example_list = images_info_of_space
						for example in tqdm(example_list):
								if example['state'] == '0' and len(empty_space_filepath) == 0:
										empty_space_filepath = example['filepath']
										img_empty_space = cv2.imread(empty_space_filepath)
										break
						for example in tqdm(example_list):
								comparision_space_filepath = example['filepath']
								img_comparission_space = cv2.imread(comparision_space_filepath)
								try_opencv_method(img_empty_space, img_comparission_space, example['state'])

if __name__ == "__main__":
		main()
