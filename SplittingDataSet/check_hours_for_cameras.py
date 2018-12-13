"""
Print the hours in wich the images where taken by each camera
"""
from itertools import groupby
from operator import itemgetter

def extractInfoFromLine(line):
		lineSeparated = line.split("/")
		fileInfoSeparated = lineSeparated[3].split("_")
		spaceNumber = fileInfoSeparated[4].split(".")[0]
		return {
				'weather': lineSeparated[0],
				'date': lineSeparated[1],
				'hour': fileInfoSeparated[2],
				'camera': fileInfoSeparated[3],
				'space': spaceNumber,
				'state': line[-2]
		}

def groupbyLocal(list, key):
		list.sort(key=lambda x: x[key])
		return [k for k, v in groupby(list, key=lambda x: x[key])]


def groupby2(list, key1, key2):
		grouper = itemgetter(key1, key2)
		return [dict(zip([key1, key2], key)) for key, grp in groupby(sorted(list, key=grouper), grouper)]


"""
Creates a list with one to many relation from one to one relation
returns: dictionary where each represents the one, each item has a list where are the many items
"""
def relateWithList(keys, keyName, itemName, one2OneRelation):
		one2ManyRelation = {}
		for key in keys:
				one2ManyRelation[key] = []

		for i in one2OneRelation:
				one2ManyRelation.get(i.get(keyName)).append(i.get(itemName))
		return one2ManyRelation


cameraFileName = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\camera1.txt"

file = open(cameraFileName, "r")
imagesInfo = [extractInfoFromLine(line) for line in file]

print(imagesInfo[0])

dates = groupbyLocal(imagesInfo, 'date')

imageDateHour = groupby2(imagesInfo, 'date', 'hour')
hoursPerDate = relateWithList(dates, 'date', 'hour', imageDateHour)


spaces = groupbyLocal(imagesInfo, 'spaces')
imageSpaceDate = groupby2(imagesInfo, 'space', 'date')
spacesDates = relateWithList(dates, 'space', 'date', imageSpaceDate)


# Obtain the list group by date and space
imageDateSpace = groupby2(imagesInfo, 'date', 'space')
# This dicionary only has the spaces related to each date, the date is the key
spacesPerDate = relateWithList(dates, 'date', 'space', imageDateSpace)

imagesInfoReduced = []

print(spacesPerDate)

for date, hours in hoursPerDate.items():
		print("Fecha: " + date + " Num. horas: {}".format(len(date)))

cuenta = True
for date, spaces in spacesPerDate.items():
	spaces.sort()
	print("Inicio: {}".format(spaces[0]) + "Fin: {}".format(spaces[-1]))
	print("Fecha: " + date + " Num. espacios: {}".format(len(spaces)))










