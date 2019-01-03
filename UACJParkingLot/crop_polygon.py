import numpy
from PIL import Image, ImageDraw
import pickle
import json
import os
from tqdm import tqdm
def crop_image(image_path, out_image, polygon):
		#print(image_path)
		# read image as RGB and add alpha (transparency)
		# 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_12-35-19.jpg'
		im = Image.open(image_path).convert("RGBA")

		#print(polygon)
		# convert to numpy (for convenience)
		imArray = numpy.asarray(im)

		# create mask
		maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
		ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
		mask = numpy.array(maskIm)

		# assemble new image (uint8: 0-255)
		newImArray = numpy.empty(imArray.shape, dtype='uint8')

		# colors (three first columns, RGB)
		newImArray[:, :, :3] = imArray[:, :, :3]

		# transparency (4th column)
		newImArray[:, :, 3] = mask * 255

		# back to Image from numpy
		newIm = Image.fromarray(newImArray, "RGBA")
		newIm.save(out_image)

if __name__ == "__main__":
		image_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_14-49-19.jpg'
		destination_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\Splited\\30_4\\image30-11-2018_14-49-19'
		if not os.path.isdir(destination_path):
				os.mkdir(destination_path)
		# open output file for reading
		with open('map_points2.txt', 'r') as filehandle:
				polygons = json.load(filehandle)
		count = 0
		for polygon in tqdm(polygons):
				polygon = [tuple([x * 3 for x in l]) for l in polygon]
				crop_image(image_path, os.path.join(destination_path, "{}.png".format(count)), polygon)
				count += 1
		#polygon = [(1482, 767), (1453, 706), (1436, 627), (1425, 593)]

