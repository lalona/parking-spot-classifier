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
    dataset_path = 'C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot'
    path = os.path.join(dataset_path, image_info['filepath'], image_info['filename'])
    return path


def extractInfoFromFile(label_path):
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
            'filepath': space_path,
            'filename': file_name,
            'weather': dir_weather,
            'date': date,
            'hour': hour,
						'parkinglot': dir_parking_lot,
            'space': id,
            'state': occupied
        }
        # only append the info if is a valid file
        if os.path.isfile(getPathToImage(dictionary)):
            print(dictionary)
            info.append(dictionary)
        else:
            skipped += 1
            continue
    return skipped, info


path_labels = "C:\\Eduardo\\ProyectoFinal\\Datasets\\PKLot\\PKLot\\"
file_name = "pklot_labels2.txt"

def main():
    parser = argparse.ArgumentParser(description='Force the save.')
    parser.add_argument("-f", "--force", type=bool, default=False, help='If the file was already made, still make it')
    args = vars(parser.parse_args())

    force_creation = args["force"]

    if os.path.isfile(file_name) and not force_creation:
        print('The file was already created if you want to repeat the process you can set the param -f to True')
        return

    images_info = []
    file_skipped = []
    skipped = 0
    # For each .xml file extract the info from that file
    for subdir, dirs, files in os.walk(path_labels):
        for file in files:
            if file.endswith('.xml'):
                s, info = extractInfoFromFile(os.path.join(subdir, file))
                if s > 30:
                    file_skipped.append(os.path.join(subdir, file))
                skipped += s
                images_info.extend(info)
        print(subdir)
    print("Total of images info extracted: {} and a total of {} was skiped".format(len(images_info), skipped))

    # sort the info
    grouper = itemgetter('parkinglot', 'date', 'space', 'hour')
    images_info = sorted(images_info, key=grouper)

    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(images_info, fp)

    print("SAVED")

if __name__ == "__main__":
		main()




