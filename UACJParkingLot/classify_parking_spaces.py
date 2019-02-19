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
import copy
import ntpath
import os
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


# x - x1=dcos(arctan(m))
def get_x2(d, m, x1):
    return (d * math.cos(math.atan(m))) + x1


def distance_between_points(point1, point2):
    # sqrt( (x2 - x1)^2 + (y2 - y2)^2 )
    return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))


# https://math.stackexchange.com/questions/516219/finding-out-the-area-of-a-triangle-if-the-coordinates-of-the-three-vertices-are
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


def indicated_rectangle(rectangle, point):
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
    if t == rectangle_area:
        return True
    return False

M_DEFINE_SLOPES = 'define_slopes'
M_DEFINE_IDS = 'define_ids'
M_HELP = 'help'
M_MAP = 'define_map'
M_CHANGE_ANGLE = 'change_angle'

class DefineStates():
    def __init__(self, image, image_path, map=[], angle=0, scale=3):
        self.map = map
        self.dir_image = os.path.dirname(os.path.abspath(image_path))
        self.image_path = image_path
        self.states = []
        if os.path.isfile(self.getStatesFilename()):
            with open(self.getStatesFilename(), 'r') as filehandle:
                self.states = json.load(filehandle)
        self.angle = 0
        self.angle_count = 0

        height, width, channels = image.shape
        self.imS = cv2.resize(image, (int(width / scale), int(height / scale)))
        self.imSrotation = self.imS.copy()
        self.imSHelp = self.imS.copy()
        self.imSclean = self.imS.copy()
        self.imSclean_rotated = self.imS.copy()
        self.mode = M_MAP

        if len(self.map) > 0:
            if angle == -1:
                self.angle = self.map[0]["angle"]
            else:
                self.angle = angle
            self.rotateImage(clean=True)
            self.drawMap(clean=True)
        else:
            self.drawMap(clean=True)

    def drawMap(self, clean=False):
        if clean:
            self.imS = self.imSclean_rotated
            self.imSclean_rotated = self.imSclean_rotated.copy()
        count = 1
        for info in self.map:
            if self.angle != info["angle"]:
                continue
            m = info["rectangle_scaled"]

            has_state = False
            for s in self.states:
                if s['id'] == info['id']:
                    if s['state'] == '0':
                        cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
                                      (int(m[1][0]), int(m[1][1])), (255, 0, 0), 1)
                    else:
                        cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
                                      (int(m[1][0]), int(m[1][1])), (255, 255, 255), 1)
                    has_state = True
                    break
            if has_state:
                continue

            if count % 2 == 0:
                cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
                              (int(m[1][0]), int(m[1][1])), (0, 255, 0), 1)
            else:
                cv2.rectangle(self.imS, (int(m[3][0]), int(m[3][1])),
                              (int(m[1][0]), int(m[1][1])), (255, 255, 0), 1)
            count += 1
            if self.mode == M_DEFINE_IDS:
                center = center_of_rectangle(m)
                center = (int(center[0]), int(center[1]))
                id = '{}'.format(info['id'])
                cv2.putText(self.imS, id, center, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(self.imS, "{}".format(self.angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    def rotateImage(self, clean=True):
        self.imS = self.imSclean
        self.imSclean = self.imSclean.copy()
        self.imS = imutils.rotate(self.imS, self.angle)
        self.imSclean_rotated = self.imS
        self.drawMap(clean=clean)

    def handleLeftClick(self, point):
        self.addState(point, '0')

    def handleRightClick(self, point):
        self.addState(point, '1')

    def addState(self, point, state):
        indicated_ = None
        for m in reversed(self.map):
            if m['angle'] == self.angle:
                if indicated_rectangle(m['rectangle_scaled'], point):
                    indicated_ = m
                    break
        if indicated_ is not None:
            changed = False
            for s in self.states:
                if s['id'] == indicated_['id']:
                    s['state'] = state
                    changed = True
                    break
            if not changed:
                indicated_state = copy.deepcopy(indicated_)
                indicated_state.pop('rectangle_scaled')
                indicated_state['state'] = state
                self.states.append(indicated_state)
        print(self.states)
        self.drawMap(clean=True)


    def changeAngle(self):
        self.angle_count += 1
        angles = extract_unique_items_by_key(self.map, "angle")
        if self.angle_count >= len(angles):
            self.angle_count = 0
        if len(angles) > 0:
            self.angle = angles[self.angle_count]
            self.rotateImage(self.angle)
            self.drawMap()

    def setMode(self, mode, default=M_MAP):
        if mode == self.mode:
            self.mode = default
        else:
            self.mode = mode

    def showImage(self):
        if self.mode == M_HELP:
            cv2.imshow("image", self.imSHelp)
        else:
            cv2.imshow("image", self.imS)

    def showHelp(self, functionalaty):
        self.setMode(M_HELP)
        if self.mode == M_HELP:
            self.imSHelp = self.imSclean
            self.imSclean = self.imSclean.copy()
            initial_height = 30
            for f in functionalaty:
                help = '{}: {}'.format(f['key'], f['function'])
                cv2.putText(self.imSHelp, help, (10, initial_height), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 153, 0), 1, cv2.LINE_AA)
                initial_height += 30

    def repeatStates(self, last_states):
        self.states = last_states
        self.drawMap(clean=True)

    def saveStates(self):
        with open(self.getStatesFilename(), 'w') as filehandle:
            json.dump(self.states, filehandle)

    def getStatesFilename(self):
        image_filename = ntpath.basename(self.image_path).split('.')[0]
        image_filename = image_filename + '_states.json'
        return os.path.join(self.dir_image, image_filename)

def click_and_crop(event, x, y, flags, param):
    define_states = param
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        define_states.handleLeftClick(point)
    if event == cv2.EVENT_RBUTTONDOWN:
        point = (x, y)
        define_states.handleRightClick(point)


def setFunctionality(define_states, last_states):
    functionality = [
        {'key': 'a', '_callback': define_states.changeAngle, 'function': 'Cambia la imagen al siguiente angulo en el mapa.'},
        {'key': 'h', '_callback': define_states.showHelp, 'function': 'Muestra la funcionalidad.'},
        {'key': 's', '_callback': define_states.saveStates, 'function': 'Guarda los estados.'},
        {'key': 'c', '_callback': define_states.repeatStates, 'param': last_states, 'function': 'Pone los estados igual que la imagen pasada.'},
        {'key': 'b', 'function': 'Muestra la imagen anterior.'},
        {'key': 'n', 'function': 'Muestra la siguiente imagen.'},
        {'key': 'x', 'function': 'Guarda el mapa y sale del programa.'},
    ]
    functionality[1]['param'] = functionality
    return functionality

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scale", required=False, default=3, help="The scale where the image can be better managed")
    ap.add_argument("-i", "--image", required=False, default=3, help="Initial image")
    args = vars(ap.parse_args())
    scale = float(args['scale'])
    initial_image = args['image']

    uacj_images_path = 'C:\\Eduardo\\tesis\\datasets\\uacj'
    # uacj_images_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ'
    images_path = get_images_path(uacj_images_path)
    if initial_image is not None:
        try:
            current_image_i = images_path.index(initial_image)
        except:
            print('The image given doesnt exist')
            return
    else:
        current_image_i = 0
    image_path = images_path[current_image_i]
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(image_path)
    cv2.namedWindow("image")


    map_file = 'map_points_crop_rectangle_2.txt'
    map = []
    if os.path.isfile(map_file):
        with open(map_file, 'r') as filehandle:
            map1 = json.load(filehandle)
        for info in map1:
            info["rectangle"] = [tuple([x for x in l]) for l in info["rectangle"]]
            info["angle"] = float(info["angle"])
            map.append(info)
    for m in map:
        rectangle = m['rectangle']
        fixed_rectangle = []
        for p in rectangle:
            p = (int(p[0] / scale), int(p[1] / scale))
            fixed_rectangle.append(p)
        m['rectangle_scaled'] = fixed_rectangle

    define_states = DefineStates(image=image, image_path=image_path, map=map, scale=scale, angle=-1)
    cv2.setMouseCallback("image", click_and_crop, define_states)

    functionality = setFunctionality(define_states, [])

    # keep looping until the 'q' key is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF

        for f in functionality:
            if key == ord(f['key']):
                if '_callback' in f:
                    if 'param' in f:
                        f['_callback'](f['param'])
                    else:
                        f['_callback']()
        if key == ord("b"):
            current_image_i -= 1
            if current_image_i < 0:
                current_image_i = len(images_path) - 1
            image_path = images_path[current_image_i]
            image = cv2.imread(image_path)
            define_states.saveStates()
            last_states = define_states.states
            define_states = DefineStates(image=image, image_path=image_path, map=map, angle=define_states.angle, scale=scale)
            functionality = setFunctionality(define_states, last_states)
            cv2.setMouseCallback("image", click_and_crop, define_states)
        if key == ord("n"):
            current_image_i += 1
            if current_image_i >= len(images_path):
                current_image_i = 0
            image_path = images_path[current_image_i]
            print(image_path)
            image = cv2.imread(image_path)
            define_states.saveStates()
            last_states = define_states.states
            define_states = DefineStates(image=image, image_path=image_path, map=map, angle=define_states.angle, scale=scale)
            functionality = setFunctionality(define_states, last_states)
            cv2.setMouseCallback("image", click_and_crop, define_states)
        if key == ord("x"):
            break

        define_states.showImage()

    define_states.saveStates()

if __name__ == "__main__":
    main()

