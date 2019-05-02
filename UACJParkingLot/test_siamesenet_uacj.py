"""
Esta es la culminacion de todos los esfuerzos
"""

# import the necessary packages
import argparse
import cv2
import imutils
from random_image import get_images_path
from PIL import Image
import argparse
import os
import json
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import ntpath
import copy
import sys
from keras.preprocessing.image import load_img, img_to_array

sys.path.insert(0, "C:\\Eduardo\\tesis\\parking-spot-classifier\\MyOwn_siamesenet\\my_siamesenetwork_tensorflow\\refactored")
from all_psiamesenets import getModel

class TestAllImages():
    def __init__(self, list_images, model_name, map_file, model_path, net, time_step=5):
        self.model_path = model_path
        self.net = net
        self.model_input_dim = 224
        self.list_images = list_images
        self.model_name = model_name
        self.time_step = time_step
        self.map_file = map_file
        self.map = []
        if os.path.isfile(map_file):
            with open(map_file, 'r') as filehandle:
                map1 = json.load(filehandle)
            for info in map1:
                info["rectangle"] = [tuple([x for x in l]) for l in info["rectangle"]]
                info["angle"] = float(info["angle"])
                self.map.append(info)

    def getStatesFilename(self, image_path):
        dir_image = os.path.dirname(os.path.abspath(image_path))
        image_filename = ntpath.basename(image_path).split('.')[0]
        image_filename = image_filename + '_states.json'
        return os.path.join(dir_image, image_filename)

    def getSpaceImg(self, orig_img, rectangle):
        height, width, channels = orig_img.shape
        initial_x = width
        final_x = 0
        initial_y = height
        final_y = 0
        for point in rectangle:
            x = point[0]
            y = point[1]
            if x < initial_x:
                initial_x = x
            if x > final_x:
                final_x = x
            if y < initial_y:
                initial_y = y
            if y > final_y:
                final_y = y
        space_img = orig_img[int(initial_y):int(final_y), int(initial_x):int(final_x)]
        img = Image.fromarray(space_img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width_height_tuple = (self.model_input_dim, self.model_input_dim)
        if img.size != width_height_tuple:
            img = img.resize(width_height_tuple, Image.NEAREST)
        x = img_to_array(img)
        return x

    def getCompVectorEveryTimeStep(self, img_parking, space, sess, img_placeholder, net):
        last_state = space['last_state']
        state = space['state']
        count = space['count']
        if last_state == state and count < self.time_step:
            space['count'] += 1
            return []
        else:
            last_state = state
            space['count'] += 0
            space['last_state'] = last_state
            return self.getCompVector(img_parking, space, sess, img_placeholder, net)

    def getSpaceInfo(self, space, img_path):
        return {
            'img_path': img_path,
            'space_id': space['id']
        }

    def addInfo(self, pred, errors_info, info, space, day, state):
        #print(errors_info['by_space'][space][pred])
        errors_info['by_space'][space][pred].append(info)
        errors_info['by_date'][day][pred].append(info)
        errors_info['by_state'][state][pred].append(info)

    def getCompVector(self, img_parking, space, sess, img_placeholder, net):
        if space['angle'] > 0:
            img_parking_rotated = imutils.rotate(img_parking, space['angle'])
        else:
            img_parking_rotated = img_parking
        space_img = self.getSpaceImg(img_parking_rotated, space['rectangle'])
        idxs_left_empty = []
        idxs_left_empty.append(space_img)
        left_empty = np.array(idxs_left_empty) / 255.0
        comp_vect = sess.run(net, feed_dict={img_placeholder: left_empty})
        return comp_vect

    def addComparissionVectorsToMap(self, img_placeholder, net, sess):
        """
        Agrega en el mapa los vectores de comparacion para espacios vacios y ocupados
        :return:
        """
        for img_path in tqdm(self.list_images):
            if os.path.isfile(self.getStatesFilename(img_path)):
                with open(self.getStatesFilename(img_path), 'r') as filehandle:
                    states = json.load(filehandle)
            img_parking = cv2.imread(img_path)
            for space in self.map:
                state_count = 0
                for s in states:
                    if s['id'] == space['id']:
                        space['state'] = s['state']
                        break
                    state_count += 1
                states.pop(state_count)
                if int(space['state']) == 0:
                    if 'e_comp' not in space:
                        vect_comp = self.getCompVector(img_parking, space, sess, img_placeholder, net)
                        space['e_comp'] = vect_comp
                else:
                    if 'o_comp' not in space:
                        vect_comp = self.getCompVector(img_parking, space, sess, img_placeholder, net)
                        space['o_comp'] = vect_comp

            if self.mapHasComparissionVector():
                return True

        return self.mapHasComparissionVector()


    def mapHasComparissionVector(self):
        count = 0
        for space in self.map:
            count += 1
            if 'e_comp' not in space or 'o_comp' not in space:
                print(space)
                return False

        return True

    def testAccuracy(self):
        failed = 0
        count = 0
        errors_info = {'error': 0, 'map_file': self.map_file, 'time_step': self.time_step, 'by_space': {}, 'by_state': {}, 'by_date': {}}
        self.addKeyErrorInfo('0', errors_info['by_state'])
        self.addKeyErrorInfo('1', errors_info['by_state'])

        img_placeholder = tf.placeholder(tf.float32, [None, self.model_input_dim, self.model_input_dim, 3], name='img')
        model = getModel(self.net)
        net = model(img_placeholder, reuse=False)
        saver = tf.train.Saver()
        for space in self.map:
            space['last_state'] = ''
            space['count'] = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)
            self.addComparissionVectorsToMap(img_placeholder, net, sess)
            for img_path in tqdm(self.list_images):
                if os.path.isfile(self.getStatesFilename(img_path)):
                    with open(self.getStatesFilename(img_path), 'r') as filehandle:
                        states = json.load(filehandle)
                img_filename = os.path.split(img_path)[1]
                if 'image28' in img_filename:
                    day = '28'
                elif 'image29' in img_filename:
                    day = '29'
                elif 'image30' in img_filename:
                    day = '30'

                self.addKeyErrorInfo(day, errors_info['by_date'])
                img_parking = cv2.imread(img_path)


                for space in self.map:
                    state_count = 0
                    for s in states:
                        if s['id'] == space['id']:
                            space['state'] = s['state']
                            break
                        state_count += 1
                    states.pop(state_count)

                    comp_vect = self.getCompVectorEveryTimeStep(img_parking, space, sess, img_placeholder, net)
                    if len(comp_vect) > 0:
                        count += 1
                        empty_comp = space['e_comp']
                        ocuppied_comp = space['o_comp']
                        dist_empty = cdist(empty_comp, comp_vect, 'euclidean')
                        dist_occupied = cdist(ocuppied_comp, comp_vect, 'euclidean')
                        info = self.getSpaceInfo(space, img_path)
                        self.addKeyErrorInfo(space['id'], errors_info['by_space'])
                        #print(errors_info)
                        if (int(space['state']) == 0) != (dist_empty <= dist_occupied):
                            self.addInfo('failed', errors_info, info, space=space['id'], day=day, state=space['state'])
                            failed += 1
                        else:
                            self.addInfo('correct', errors_info, info, space=space['id'], day=day, state=space['state'])
                        space.pop('empty_pred', None)
                        space.pop('occupied_pred', None)
                    if (count + 1) % 50 == 0:
                        print('total: {} failed: {} error: {}'.format(count, failed, failed * 100 / count))
        errors_info['error'] = failed * 100 / count
        print('Error: {}'.format(errors_info['error']))
        return errors_info

    def addKeyErrorInfo(self, key, errors_info_by):
        if key not in errors_info_by:
            errors_info_by[key] = {'correct': [], 'failed': []}

    def rotate_image(self, angle, clean=True):
        self.imS = self.imSclean
        self.imSclean = self.imSclean.copy()
        self.imS = imutils.rotate(self.imS, self.angle)
        self.imSclean_rotated = self.imS
        self.draw_map_(clean=clean)






def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--map", required=False, help="Path to the map of the parking lot")
    ap.add_argument("-mo", "--model", required=False, help="Path to model")

    args = vars(ap.parse_args())
    map_file = args['map']
    model_path = args['model']

    model_path = os.path.join(model_path, 'model.ckpt')
    model_dir, model_filename = os.path.split(model_path)

    uacj_images_path = 'C:\\Eduardo\\tesis\\datasets\\uacj'

    error_filename = 'uacj_{}_test.json'.format(os.path.splitext(model_filename)[0])
    if not os.path.isdir(os.path.join(model_dir, 'test_info')):
        os.mkdir(os.path.join(model_dir, 'test_info'))

    error_path = os.path.join(model_dir, 'test_info', error_filename)

    if os.path.isfile(error_path):
        print('This test was already made')
        return

    images_path = get_images_path(uacj_images_path)
    test_all_images = TestAllImages(images_path, model_filename, map_file, model_path, 'psv1', time_step=5)
    error_info = test_all_images.testAccuracy()

    with open(error_path, 'w') as efile:
        json.dump(error_info, efile)


if __name__ == "__main__":
    main()

