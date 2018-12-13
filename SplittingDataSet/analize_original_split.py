def extractInfoFromLine(line):
		lineSeparated = line.split("/")
		fileInfoSeparated = lineSeparated[3].split("_")
		spaceNumber = fileInfoSeparated[4].split(".")[0]
		return {
				'filename': lineSeparated[3][:-3],
				'weather': lineSeparated[0],
				'date': lineSeparated[1],
				'hour': fileInfoSeparated[2],
				'camera': fileInfoSeparated[3],
				'space': spaceNumber,
				'state': line[-2]
		}


testFileName = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\test.txt"
trainFileName = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\train.txt"
valFileName = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\LABELS\\val.txt"

file = open(testFileName, "r")
imagesTestInfo = [extractInfoFromLine(line) for line in file]

file = open(trainFileName, "r")
imagesTrainInfo = [extractInfoFromLine(line) for line in file]

file = open(valFileName, "r")
imagesValInfo = [extractInfoFromLine(line) for line in file]


print("TRAIN  Size: {}".format(len(imagesTrainInfo)))
print("VAL   Size: {}".format(len(imagesValInfo)))
print("TEST Size: {}".format(len(imagesTestInfo)))
print("TOTAL Size: {}".format(len(imagesTrainInfo) + len(imagesValInfo)+ len(imagesTestInfo)))

# cuenta = 0
# for imageInfo in imagesTestInfo:
# 		print(imageInfo)
# 		if cuenta > 25:
# 				break
# 		cuenta += 1



