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
import imutils
from itertools import groupby
from random_image import get_images_path
import random
# 'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_14-15-19.jpg'
# 'C:\Eduardo\ProyectoFinal\Pruebas_UACJ\30_4\image30-11-2018_12-35-19.jpg'

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
def slope(point1, point2):
		return (point2[1] - point1[1]) / (point2[0] - point1[0])


def center_of_rectangle(points):
		x = [p[0] for p in points]
		y = [p[1] for p in points]
		centroid = (sum(x) / len(points), sum(y) / len(points))
		return centroid


def get_b(point, m):
		return point[1] - (point[0] * m)  # b = y - mx


def get_h(m, b, x, y):
		return y - ((m * x) + b)  # y - f1(x)


"""
The separation distance from the paralel lines
params: 
	m the slope of the line in between
	h the distance iw
"""


def distance_between_lines(m, h):
		return math.cos(math.atan(m)) * h

"""
:param point shoulb be a tuple whith (x, y)
:param m is the slope of the line
"""
def distance_from_point_to_line(point, m, b1):
		a = -m
		b = 1
		c = -b1
		x = point[0]
		y = point[1]
		print("a: {} b: {} c: {} x: {} y: {}".format(a, b, c, x, y))
		d = abs(a * x + b * y + c) / math.sqrt(math.pow(a, 2) + math.pow(b, 2))
		return d

def get_x(d, m, x1):
		return (d * (m / (math.sqrt(1 + math.pow(m, 2))))) + x1

#x - x1=dcos(arctan(m))
def get_x2(d, m, x1):
		return (d * math.cos(math.atan(m))) + x1


def distance_between_points(point1, point2):
		# sqrt( (x2 - x1)^2 + (y2 - y2)^2 )
		return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))

#https://math.stackexchange.com/questions/516219/finding-out-the-area-of-a-triangle-if-the-coordinates-of-the-three-vertices-are
def area_triangle(a, b, c):
		return 0.5 * abs((a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]))

"""
If a point was mark inside a rectangle from the map then
it will return the index of that rectangle in the list
if not is going return -1
This page contains the matematic explanation
https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
"""
def indicated_rectangle(map, point):
		count = 0
		for rectangle in map:
				rectangle_area = polygon_area(rectangle)
				A = rectangle[0]
				B = rectangle[1]
				C = rectangle[2]
				D = rectangle[3]
				# △APD,△DPC,△CPB,△PBA
				t1 = area_triangle(A, point, D)
				t2 = area_triangle(D, point, C)
				t3 = area_triangle(C, point, B)
				t4 = area_triangle(point, B, A)
				t = t1 + t2 + t3 + t4
				print("t: {} rectanlge_area: {}".format(t, rectangle_area))
				if t == rectangle_area:
						return count
				count += 1
		return -1


def extract_unique_items_by_key(list, key):
		"""
		This will take a list and sorted by the key and
		then it will return the list with just the elements from that
		key without duplicates
		:param list:
		:param key:
		:return:
		"""
		list.sort(key=lambda x: x[key])
		return [k for k, v in groupby(list, key=lambda x: x[key])]


# https://plot.ly/python/polygon-area/
def polygon_area(corners):
		n = len(corners)
		area = 0.0
		for i in range(n):
				j = (i + 1) % n
				area += corners[i][0] * corners[j][1]
				area -= corners[j][0] * corners[i][1]
		area = abs(area) / 2.0
		return area

class CropRectangles():
		def __init__(self, image, map = [], angle = 0):
				self.refPt = []
				self.count = 0
				self.map = map
				self.define_slopes = False
				self.f1_captured = False
				self.draw_map = False
				self.remove = False
				self.rectangle = -1
				self.substitute = False

				height, width, channels = image.shape
				self.imS = cv2.resize(image, (int(width / 3), int(height / 3)))
				self.imS_clean = self.imS.copy()
				self.imScopy = self.imS.copy()
				self.clean = False
				self.draw_map = True
				self.angle = 0
				self.angle_count = 0

				if len(self.map) > 0:
						if angle == 0:
							self.angle = self.map[0]["angle"]
						else:
							self.angle = angle
						self.rotate_image(self.angle)
				self.draw_map_(self.map)

		def draw_map_(self, map, clean = True):
				if clean:
						self.imS = self.imS_clean
						self.imS_clean = self.imS.copy()
				count = 1
				for info in map:
						if self.angle != info["angle"]:
								continue
						m = info["rectangle"]
						if count % 2 == 0:
								cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
															(int(m[1][0]), int(m[1][1])), (0, 255, 0), 2)
						else:
								cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
															(int(m[1][0]), int(m[1][1])), (255, 255, 0), 2)
						count += 1
				cv2.putText(self.imS, "{}".format(self.angle), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

		def rotate_image(self, angle):
				self.imS = self.imScopy
				self.imScopy = self.imS.copy()
				self.imS = imutils.rotate(self.imS, self.angle)
				self.imS_clean = self.imS.copy()

		def handle_left_click(self, point):
				if self.define_slopes:
						if self.count == 0:
								self.refPt = [point]
						elif self.count == 1:
								m = slope(self.refPt[0], point)
								self.angle = math.degrees(math.atan(m))
								self.rotate_image(self.angle)
								self.define_slopes = False
								self.count = -1
						self.count += 1
				else:
						if self.count == 0:
								self.refPt = [point]
						elif self.count == 1:
								d = distance_between_points(self.refPt[0], point)
								self.refPt.append((int(self.refPt[0][0] + d), self.refPt[0][1]))
						elif self.count == 2:
								self.refPt.append((self.refPt[1][0], point[1]))
								self.refPt.append((self.refPt[0][0], point[1]))
								space = {"angle": self.angle, "rectangle": self.refPt, "id": len(self.map)}
								self.map.append(space)
								self.draw_map_(self.map)
								self.count = -1
						self.count += 1

		def handle_right_click(self):
				if len(self.map) > 0 and self.draw_map:
						self.map = self.map[:-1]
						self.refPt = []
						self.count = 0
						self.draw_map_(self.map)

		def change_angle(self):
				self.angle_count += 1
				angles = extract_unique_items_by_key(self.map, "angle")
				print(angles)
				if self.angle_count >= len(angles):
						self.angle_count = 0
				if len(angles) > 0:
						self.angle = angles[self.angle_count]
						self.rotate_image(self.angle)
						self.draw_map_(self.map)

		def increment_angle(self):
				self.angle += 5
				self.rotate_image(self.angle)
				self.draw_map_(self.map)

		def decrement_angle(self):
				self.angle -= 5
				self.rotate_image(self.angle)
				self.draw_map_(self.map)

		def reset_angle(self):
				self.angle = 0
				self.rotate_image(self.angle)
				self.draw_map_(self.map)
				self.angle_count = -1

		def set_mode_define_slopes(self):
				self.count = 0
				self.define_slopes = not self.define_slopes

		def show_image(self):
				if self.define_slopes:
						cv2.imshow("image", self.imScopy)
				else:
						cv2.imshow("image", self.imS)

def click_and_crop(event, x, y, flags, param):
		global crop_rectangle
		if event == cv2.EVENT_LBUTTONDOWN:
				point = (x, y)
				crop_rectangle.handle_left_click(point)

		if event == cv2.EVENT_RBUTTONDOWN:
				crop_rectangle.handle_right_click()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image_path = os.path.join('C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ', args["image"])
images_path = get_images_path()
image_path = images_path[0]
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(image_path)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

map_file = 'map_points_crop_rectangle_2.txt'
map = []
if os.path.isfile(map_file):
		with open(map_file, 'r') as filehandle:
				map1 = json.load(filehandle)
		for info in map1:
			info["rectangle"] = [tuple([x for x in l]) for l in info["rectangle"]]
			info["angle"] = float(info["angle"])
			map.append(info)

crop_rectangle = CropRectangles(image=image, map=map)

# keep looping until the 'q' key is pressed
while True:
		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
				break

		if key == ord("a"):
			crop_rectangle.change_angle()
		if key == ord("i"):
				crop_rectangle.increment_angle()
		if key == ord("o"):
				crop_rectangle.decrement_angle()
		if key == ord("r"):
			crop_rectangle.reset_angle()
		if key == ord("s"):
			crop_rectangle.set_mode_define_slopes()
		if key == ord("j"):
			image_path = random.choice(images_path)
			# load the image, clone it, and setup the mouse callback function
			image = cv2.imread(image_path)
			crop_rectangle = CropRectangles(image, map, crop_rectangle.angle)

		crop_rectangle.show_image()

# open output file for writing
with open(map_file, 'w') as filehandle:
		json.dump(map, filehandle)

# if there are two reference points, then crop the region of interest
# from teh image and display it

# define list with values

# open output file for writing

# if len(refPt) == 4:
# 	cv2.destroyAllWindows()
# 	roi = crop_image(image_path, refPt)
# 	#with open(r"polygon.pickle", "wb") as output_file:
# 	#		pickle.dump(refPt, output_file)
# 	cv2.waitKey(0)
