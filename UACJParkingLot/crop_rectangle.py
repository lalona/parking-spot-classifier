import imutils
import cv2
import json
import math
import os
from tqdm import tqdm
def slope(point1, point2):
		if (point2[0] - point1[0]) == 0:
				print("Warning: division by zero. Returning 0 as slope")
				return 0
		return (point2[1] - point1[1]) / (point2[0] - point1[0])

# https://gamedev.stackexchange.com/questions/86755/how-to-calculate-corner-positions-marks-of-a-rotated-tilted-rectangle
def rotate_rectangle(x, y, cx, cy, angle):
	temp_x = x - cx
	temp_y = y - cy
	rotated_x = temp_x * math.cos(angle) - temp_y * math.sin(angle)
	rotated_y = temp_x * math.sin(angle) + temp_y * math.cos(angle)

	x = (rotated_x * -1) + cx
	y = (rotated_y * -1) + cy
	#print("rotated_x {} rotated_y {}".format(rotated_x, rotated_y))
	return x, y

# https://stackoverflow.com/questions/4355894/how-to-get-center-of-set-of-points-using-python
def center_of_rectangle(points):
		x = [p[0] for p in points]
		y = [p[1] for p in points]
		centroid = (sum(x) / len(points), sum(y) / len(points))
		return centroid

def up_y(rectangle, max_y):
		y = max_y
		for point in rectangle:
				if point[1] < y:
						y = point[1]
		return y


def down_y(rectangle, min_y = 0):
		y = min_y
		for point in rectangle:
				if point[1] > y:
						y = point[1]
				elif point[1] < 0 and y == 0:
						y = point[1]
		return y


def left_x(rectangle, max_x):
		x = max_x
		for point in rectangle:
				if point[0] < x:
						x = point[0]
		return x


def right_x(rectangle):
		x = 0
		for point in rectangle:
				if point[0] > x:
						x = point[0]
		return x

if __name__ == "__main__":
		image_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_14-49-19.jpg'
		destination_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\Splited\\30_4\\image30-11-2018_14-49-19'
		if not os.path.isdir(destination_path):
				os.mkdir(destination_path)

		image = cv2.imread(image_path)
		with open('map_points3.txt', 'r') as filehandle:
				polygons = json.load(filehandle)
		map = []
		for polygon in polygons:
				map.append([tuple([x for x in l]) for l in polygon])
		height, width, channels = image.shape
		imS = cv2.resize(image, (int(width / 3), int(height / 3)))
		height, width, channels = imS.shape
		count = 0
		for m in tqdm(map):
				polygon = [tuple([x for x in l]) for l in m]
				m = slope(polygon[0], polygon[3])
				angle = math.degrees(math.atan(m))
				angleRad = math.radians(180 - angle)
				rotated = imutils.rotate(imS, angle)
				centroid = center_of_rectangle(polygon)
				#rotate_rectangle(p[0], p[1], centroid[0], centroid[1], angleRad)
				#rotated_polygon = [rotate_rectangle(p[0], p[1], centroid[0], centroid[1], angleRad) for p in polygon]
				rotated_polygon = [rotate_rectangle(p[0], p[1], int(width / 2), int(height / 2), angleRad) for p in polygon]
				#print("polygon: {} centroid: {} angle: {} rotated_polygon: {}".format(polygon, centroid, angleRad, rotated_polygon))
				#print("y: {} y + h: {} x: {} x + w: {}".format(int(rotated_polygon[3][1]), int(rotated_polygon[3][1] + (rotated_polygon[2][1] - rotated_polygon[3][1])), int(rotated_polygon[3][0]), int(rotated_polygon[3][0] + (rotated_polygon[0][0] - rotated_polygon[3][0]))))

				#crop_img = rotated[int(rotated_polygon[2][1]):int(rotated_polygon[2][1] + (rotated_polygon[3][1] - rotated_polygon[2][1])), int(rotated_polygon[2][0]):int(rotated_polygon[3][0] + (rotated_polygon[0][0] - rotated_polygon[2][0]))]
				up = up_y(rotated_polygon, height)
				left = left_x(rotated_polygon, width)
				crop_img = rotated[int(up):int(up + (down_y(rotated_polygon) - up)), int(left):int(left + (right_x(rotated_polygon) - left))]
				cv2.rectangle(rotated, (int(rotated_polygon[2][0]), int(rotated_polygon[2][1])), (int(rotated_polygon[0][0]), int(rotated_polygon[0][1])), (0, 255, 0), 2)
				try:
					cv2.imshow("Rotated (Problematic)", rotated)
					cv2.imwrite(os.path.join(destination_path, "{}.jpg".format(count)), crop_img)
				except:
					print("up_y: {} down_y: {} left_x: {} right_x: {}".format(up, down_y(rotated_polygon), left, right_x(rotated_polygon)))
					print(rotated_polygon)
				cv2.waitKey(0)
				count += 1
