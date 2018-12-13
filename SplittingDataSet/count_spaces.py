from os import listdir
from os.path import isfile, join
mypath = "C:\\Eduardo\\ProyectoFinal\\Datasets\\CNR-EXT\\PATCHES\\OVERCAST\\2015-11-16\\camera1\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

spaceInitialRange = -11
spaceEndRange = -4

initialRange = -17
endRange = -12

onlyfilesSpaceNames = list(set([float(f[initialRange:endRange]) for f in onlyfiles]))

print(onlyfilesSpaceNames)

print(len(onlyfilesSpaceNames))

