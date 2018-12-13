# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
from crop_polygon import crop_image
import pickle
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
count = 0


def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, count

		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
				print((x,y))
				if count == 0:
						refPt = [(x, y)]
				elif count < 4:
						refPt.append((x, y))
				count += 1

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image_path = args["image"]
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(image_path)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 4:
	cv2.destroyAllWindows()
	roi = crop_image(image_path, refPt)
	#with open(r"polygon.pickle", "wb") as output_file:
	#		pickle.dump(refPt, output_file)
	cv2.waitKey(0)


