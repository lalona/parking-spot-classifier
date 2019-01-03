# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
from crop_polygon import crop_image
import json
import math
import numpy as np
import os
# 'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_14-15-19.jpg'
# 'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_12-35-19.jpg'

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
count = 0
map = []
line1 = []
line2 = []
m1 = 0
b1 = 0
m2 = 0
b2 = 0
define_slopes = True
f1_captured = False
draw_map = False

def slope(point1, point2):
		return (point2[1] - point1[1]) / (point2[0] - point1[0])


def get_b(point, m):
		return point[1] - (point[0] * m)  # b = y - mx


def get_h(m, b, x, y):
		return y - ((m * x) + b)  # y - f1(x)


def distance(m, h):
		return math.cos(math.atan(m)) * h


def get_x(d, m, x1):
		return (d * (m / (math.sqrt(1 + math.pow(m, 2))))) + x1

#x - x1=dcos(arctan(m))
def get_x2(d, m, x1):
		parte1 = d * math.cos(math.atan(m))
		print("Parte1: {} x1: {}".format(parte1, x1))
		return (d * math.cos(math.atan(m))) + x1


def distance_between_points(point1, point2):
		# sqrt( (x2 - x1)^2 + (y2 - y2)^2 )
		return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))

def draw_map2():
		global map, imS_map, draw_map
		for m in map:
				pts = np.array(m, np.int32)
				pts = pts.reshape((-1, 1, 2))
				cv2.polylines(imS_map, [pts], True, (0, 255, 255))
		draw_map = True

def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, count, map, imS_slopes, imS_lines, imS_map, imScopy, imS, m1, b1, m2, b2, f1_captured, line1, line2, define_slopes, draw_map

		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
				point = (x, y)
				if define_slopes:
						cv2.circle(imS_slopes, point, 4, (0, 255, 0), -1)
						cv2.circle(imS, point, 4, (0, 255, 0), -1)
						if count == 0:
								refPt = [point]
								count = 1
						elif count == 1:
								refPt.append((x, y))
								cv2.line(imS_slopes, refPt[0], refPt[1], (255, 0, 0), 2)
								if not f1_captured:
										m1 = slope(refPt[0], refPt[1])
										print(m1)
										b1 = get_b(refPt[0], m1)
										line1 = refPt
										f1_captured = True
								else:
										m2 = slope(refPt[0], refPt[1])
										b2 = get_b(refPt[0], m2)
										line2 = refPt
										define_slopes = False
								count = 0

				else:
						if count == 0:
								refPt = [point]
								draw_map = False
								cv2.circle(imS_lines, point, 4, (0, 255, 0), -1)
								count += 1
						elif count == 1:
								refPt.append(point)
								cv2.circle(imS_lines, point, 4, (0, 255, 0), -1)
								b = get_b(refPt[0], m1)
								d = distance_between_points(refPt[0], refPt[1])
								x1 = get_x2(d, math.fabs(m1), refPt[0][0])
								y1 = (m1 * x1) + b
								refPt[1] = (int(x1), int(y1))
								m1 = slope(refPt[0], refPt[1])
								cv2.line(imS_lines, refPt[0], refPt[1], (255, 0, 0), 2)
								count += 1
						elif count == 2:
								b = get_b((x, y), m1)
								h = get_h(m1, b1, x, y)
								d = distance(m2, h)
								x1 = get_x2(d, m2, refPt[0][0])
								x2 = get_x2(d, m2, refPt[1][0])
								y1 = (m1 * x1) + b
								y2 = (m1 * x2) + b
								refPt.append((int(x2), int(y2)))
								refPt.append((int(x1), int(y1)))
								cv2.line(imS_lines, refPt[2], refPt[3], (0, 255, 0), 2)
								map.append(refPt)
								print(refPt)
								draw_map2()
								count = 0


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
imS_slopes = imS.copy()
imS_lines = imS.copy()
imS_map = imS.copy()
imScopy = imS.copy()
draw_map = False
# keep looping until the 'q' key is pressed
while True:
		if define_slopes:
			cv2.imshow("image", imS_slopes)
		elif not draw_map:
			cv2.imshow("image", imS_lines)
		else:
			cv2.imshow("image", imS_map)
		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
				break
		if key == ord("d"):
				define_slopes = True
		if key == ord("m"):
				define_slopes = False

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
