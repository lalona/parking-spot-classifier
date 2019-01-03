# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
from crop_polygon import crop_image
import json
import numpy as np
import os
#'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_14-15-19.jpg'
#'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_12-35-19.jpg'

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
count = 0
map = []
show_polygons = True

def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, count, map, imS_polygons, imScopy, imS

		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
				point = (x, y)
				cv2.circle(imS_polygons, point, 4, (0, 255, 0), -1)
				cv2.circle(imS, point, 4, (0, 255, 0), -1)
				if count == 0:
						refPt = [point]
				elif count < 4:
						refPt.append((x, y))
				if count == 3:
						map.append(refPt)
						imS_polygons = imScopy
						imScopy = imS_polygons.copy()
						for m in map:
								pts = np.array(m, np.int32)
								pts = pts.reshape((-1, 1, 2))
								cv2.polylines(imS_polygons, [pts], True, (0, 255, 255))
						print("Se agrego un espacio: {}".format(len(map)))
						count = 0
						imS = imScopy.copy()
				else:
						count += 1
		if event == cv2.EVENT_RBUTTONDOWN and len(map) > 0:
				map = map[:-1]
				imS_polygons = imScopy
				imScopy = imS_polygons.copy()
				refPt = []
				count = 0
				for m in map:
						pts = np.array(m, np.int32)
						pts = pts.reshape((-1, 1, 2))
						cv2.polylines(imS_polygons, [pts], True, (0, 255, 255))
				print("Se removio el espacio: {}".format(len(map) + 1))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image_path = os.path.join('C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ', args["image"])
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(image_path)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
height, width, channels = image.shape
imS = cv2.resize(image, (int(width / 3), int(height / 3)))
imS_polygons = imS.copy()
imScopy = imS.copy()
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	if show_polygons:
		cv2.imshow("image", imS_polygons)
	else:
		cv2.imshow("image", imS)
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
	if key == ord("s"):
		show_polygons = True
	if key == ord("h"):
		show_polygons = False

# if there are two reference points, then crop the region of interest
# from teh image and display it

# define list with values

# open output file for writing
with open('map_points2.txt', 'w') as filehandle:
    json.dump(map, filehandle)

# if len(refPt) == 4:
# 	cv2.destroyAllWindows()
# 	roi = crop_image(image_path, refPt)
# 	#with open(r"polygon.pickle", "wb") as output_file:
# 	#		pickle.dump(refPt, output_file)
# 	cv2.waitKey(0)





