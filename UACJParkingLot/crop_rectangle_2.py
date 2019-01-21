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
define_slopes = False
f1_captured = False
draw_map = False
remove = False
rectangle = -1
substitute = False
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

def draw_map2():
		global map, imS_map, draw_map
		for m in map:
				pts = np.array(m, np.int32)
				pts = pts.reshape((-1, 1, 2))
				cv2.polylines(imS_map, [pts], True, (0, 255, 255))
		draw_map = True

#https://math.stackexchange.com/questions/516219/finding-out-the-area-of-a-triangle-if-the-coordinates-of-the-three-vertices-are
def area_triangle(a, b, c):
		return 0.5 * abs((a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]))

#https://plot.ly/python/polygon-area/
def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

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

def draw_map_(map, clean = True):
		global imS, imS_clean
		if clean:
				imS = imS_clean
				imS_clean = imS.copy()
		count = 1
		for info in map:
				if angle != info["angle"]:
						continue
				m = info["rectangle"]
				if count % 2 == 0:
						cv2.rectangle(imS, (int(m[3][0]), int(m[3][1])),
													(int(m[1][0]), int(m[1][1])), (0, 255, 0), 2)
				else:
						cv2.rectangle(imS, (int(m[3][0]), int(m[3][1])),
													(int(m[1][0]), int(m[1][1])), (255, 255, 0), 2)
				count += 1
		cv2.putText(imS, "{}".format(angle), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

def rotate_image(angle):
		global imS, imScopy, imS_clean
		imS = imScopy
		imScopy = imS.copy()
		imS = imutils.rotate(imS, angle)
		imS_clean = imS.copy()


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


def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, count, map, imS_clean, imS_lines, imS_map, imScopy, imS, m1, b1, m2, b2, f1_captured, line1, line2, define_slopes, draw_map, rectangle, substitute, remove, angle

		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
				point = (x, y)
				if define_slopes:
						if count == 0:
								refPt = [point]
						elif count == 1:
								m = slope(refPt[0], point)
								angle = math.degrees(math.atan(m))
								rotate_image(angle)
								define_slopes = False
								count = -1

						count += 1
				else:
						if count == 0:
								refPt = [point]
						elif count == 1:
								d = distance_between_points(refPt[0], point)
								refPt.append((int(refPt[0][0] + d), refPt[0][1]))
						elif count == 2:
								refPt.append((refPt[1][0], point[1]))
								refPt.append((refPt[0][0], point[1]))
								print(refPt)
								space = {"angle": angle, "rectangle": refPt, "id": len(map)}
								map.append(space)
								draw_map_(map)
								count = -1
						count += 1

		if event == cv2.EVENT_RBUTTONDOWN and len(map) > 0 and draw_map:
				map = map[:-1]
				refPt = []
				count = 0
				draw_map_(map)


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
imS_clean = imS.copy()
imS2 = imS.copy()
imS_slopes = imS.copy()
imS_lines = imS.copy()
imS_map = imS.copy()
imScopy = imS.copy()
imScopy2 = imS.copy()
clean = False
draw_map = True
map_file = 'map_points_crop_rectangle_2.txt'
angle = 0
angle_count = 0
map = []
if os.path.isfile(map_file):
		with open(map_file, 'r') as filehandle:
				map1 = json.load(filehandle)
		for info in map1:
			info["rectangle"] = [tuple([x for x in l]) for l in info["rectangle"]]
			info["angle"] = float(info["angle"])
			map.append(info)

if len(map) > 0:
	angle = map[0]["angle"]
	rotate_image(angle)

draw_map_(map)

# keep looping until the 'q' key is pressed
while True:

		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
				break
		if key == ord("a"):
				angle_count += 1
				angles = extract_unique_items_by_key(map, "angle")
				if angle_count >= len(angles):
						angle_count = 0
				if len(angles) > 0:
						angle = angles[angle_count]
						rotate_image(angle)
						draw_map_(map)
		if key == ord("i"):
				angle += 5
				rotate_image(angle)
				draw_map_(map)
		if key == ord("o"):
				angle -= 5
				rotate_image(angle)
				draw_map_(map)
		if key == ord("r"):
				angle = 0
				rotate_image(angle)
				draw_map_(map)
				angle_count = -1
		if key == ord("s"):
				count = 0
				define_slopes = not define_slopes

		if define_slopes:
				cv2.imshow("image", imScopy)
		else:
				cv2.imshow("image", imS)

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
