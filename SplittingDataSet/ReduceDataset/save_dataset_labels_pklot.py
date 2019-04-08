"""
This will extract all the information of the PKLot labels an it will put it in a dictionary
in the next form:
dictionary = {
            'filepath': space_file_path,
            'filename': file_name,
            'weather': dir_weather,
            'date': date,
            'hour': hour,
            'space': id,
            'state': occupied
        }
And it will save this in a file named pklot_labels.txt
"""

import xml.etree.ElementTree as ET
import os
import pickle
import argparse
from operator import itemgetter
import constants.databases_info as c

def extractInfoFromSpace(el_space, filepath, date_hour):
    """
    For each space obtains the information of the state and the id
    With the id, state and the filepath it can create the path to the
    image, and the name of the image
    :param space:
    :return: the path and name of the image, id and the state
    """
    id = el_space.attrib['id']
    occupied = el_space.attrib['occupied']
    dir_state = "Empty"
    if occupied == '1':
        dir_state = "Occupied"

    file_path = os.path.join(filepath, dir_state)
    file_name = date_hour + "#" + str(id).zfill(3) + ".jpg"
    return file_path, file_name, id, occupied

def separateDateHour(date_hour):
    """
    This separates the date and the hour that, this have to came like 'YYYY-MM-DD_hh_mm_ss' e.g. '2012-09-12_06_05_16'
    The date is return as it is, but the hour removes the second an only lefts the hour and minute separated by a point
     'hh.mm' e.g. '06.05'
    :param date_hour: this has to be a str in the format of '[date]_[hour]' '[date]_[hour_minutes_seconds]'
    :return: the date and hour
    """
    date = date_hour.split('_')[0]
    hour_split = date_hour.split('_')[1:]
    hour = "{}.{}".format(hour_split[0],hour_split[1])
    return date, hour


def getPathToImage(image_info):
    path = os.path.join(c.pklot_path, image_info['filepath'])
    return path


def extractInfoFromFilePKLot(label_path):
    """
    This will extract the information from the file and store it in a list of dictionaries
    :param file: a xml file containing the info e.g. "C:\Eduardo\ProyectoFinal\Datasets\PKLot\PKLot\PUCPR\Rainy\2012-11-09\2012-11-09_19_37_09.xml"
    :return: a list of dictionaries with the information
    """
    directories = label_path.split('\\')
    dir_parking_lot = directories[-4]
    dir_weather = directories[-3]
    dir_date = directories[-2]
    if dir_parking_lot == "PUCPR":
        dir_parking_lot = "PUC"
    date_hour = directories[-1].split('.')[0]
    date, hour = separateDateHour(date_hour)

    el_parking_lot = ET.parse(label_path).getroot()
    spaces_path = os.path.join("PKLotSegmented", dir_parking_lot, dir_weather, dir_date)
    info = []
    skipped = 0
    for el_space in el_parking_lot:
        if 'occupied' not in el_space.attrib:
            skipped += 1
            continue
        space_path, file_name, id, occupied = extractInfoFromSpace(el_space, spaces_path, date_hour)
        dictionary = {
            c.db_json_filepath: os.path.join(space_path, file_name),
            c.db_json_weather: dir_weather,
            c.db_json_date: date,
            c.db_json_hour: hour,
            c.db_json_parkinglot_camera: dir_parking_lot,
            c.db_json_space: id,
            c.db_json_state: occupied
        }
        # only append the info if is a valid file
        if os.path.isfile(getPathToImage(dictionary)):
            print(dictionary)
            info.append(dictionary)
        else:
            skipped += 1
            continue
    return skipped, info

def getAllImagesInfoPKLot():
    images_info = []
    file_skipped = []
    skipped = 0
    # For each .xml file extract the info from that file
    for subdir, dirs, files in os.walk(c.pklot_labels_path):
        for file in files:
            if file.endswith('.xml'):
                s, info = extractInfoFromFilePKLot(os.path.join(subdir, file))
                if s > 30:
                    file_skipped.append(os.path.join(subdir, file))
                skipped += s
                images_info.extend(info)
        print(subdir)
    print("Total of images info extracted: {} and a total of {} was skiped".format(len(images_info), skipped))
    return images_info



def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-to", required=True,
                    help="The dir to save the resulting file.")
    ap.add_argument("-f", "--force", required=False, default=False,
                    help="Even if the files was already created for the creation.")

    args = vars(ap.parse_args())
    save_to_dir = args['save_to']
    force = args['force']
    file_name = c.pklot_labels_pickle

    if os.path.isfile(file_name):
        print('The file was already created, you can delete the file if you want to created again')
        if not force:
            return

    images_info = getAllImagesInfoPKLot()

    # sort the info
    grouper = itemgetter(c.db_json_parkinglot_camera, c.db_json_parkinglot_date, c.db_json_parkinglot_space,
                         c.db_json_parkinglot_hour)
    images_info = sorted(images_info, key=grouper)

    with open(os.path.join(save_to_dir, file_name), "wb") as fp:  # Pickling
        pickle.dump(images_info, fp)

    print("SAVED")

if __name__ == "__main__":
		main()




