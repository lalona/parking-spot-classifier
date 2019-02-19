from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from scipy.ndimage import imread
import numpy as np
import cv2
from skimage.measure import compare_nrmse


# specify resized image sizes
height = 200
width = 200

##
# Functions
##

def get_img(path, height=0, width=0, norm_exposure=False):
	'''
	Prepare an image for image processing tasks
	'''
	# flatten returns a 2d grayscale array
	#img = imread(path, flatten=True).astype(int)
	img = cv2.imread(path, 0)
	# resizing returns float vals 0:255; convert to ints for downstream tasks
	if height > 0 and width > 0:
		img = cv2.resize(img, (width, height))
		#img = resize(img, (height, width), preserve_range=True).astype(int)
	if norm_exposure:
		img = normalize_exposure(img)
	return img

def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)


def normalize_exposure(img):
  '''
  Normalize the exposure of an image.
  '''
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def earth_movers_distance(img_a_ne, img_b_ne):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  hist_a = get_histogram(img_a_ne)
  hist_b = get_histogram(img_b_ne)
  return wasserstein_distance(hist_a, hist_b)


def structural_sim(img_a, img_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  sim = compare_ssim(img_a, img_b)
  return sim

def nrmse(img_a, img_b):
	nrmse = compare_nrmse(img_a, img_b)
	return nrmse

def pixel_sim(img_a_ne, img_b_ne):
	'''
	Measure the pixel-level similarity between two images
	@args:
	  {str} path_a: the path to an image file
	  {str} path_b: the path to an image file
	@returns:
	  {float} a float {-1:1} that measures structural similarity
	    between the input images
	'''
	height, width = img_a_ne.shape
	return np.sum(np.absolute(img_a_ne - img_b_ne)) / (height*width) / 255


def sift_sim(img_a, img_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def mse(img_a, img_b):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
	err /= float(img_a.shape[0] * img_b.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def getMethods():
	methods = [
			{'name': 'structural_sim', 'callback': structural_sim},
			{'name': 'pixel_sim', 'callback': pixel_sim, 'normal_exposure': True},
			#{'name': 'sift_sim', 'callback': sift_sim},
			{'name': 'emd', 'callback': earth_movers_distance, 'normal_exposure': True},
			#{'name': 'mse', 'callback': mse},
			{'name': 'nrmse', 'callback': nrmse}
	]
	return methods