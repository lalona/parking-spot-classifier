import numpy
from PIL import Image, ImageDraw
import pickle

def crop_image(image_path, polygon):
		print(image_path)
		# read image as RGB and add alpha (transparency)
		# 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_12-35-19.jpg'
		im = Image.open(image_path).convert("RGBA")

		print(polygon)
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
		newIm.save("out.png")

if __name__ == "__main__":
		image_path = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30_4\\image30-11-2018_12-35-19.jpg'
		polygon = [(1482, 767), (1453, 706), (1436, 627), (1425, 593)]
		#with open(r"polygon.pickle", "rb") as input_file:
		#		polygon = pickle.load(input_file)
		print(polygon)
		crop_image(image_path, polygon)

