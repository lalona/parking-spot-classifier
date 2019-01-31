import os

def get_images_path(path_images):
		images_paths = []
		for posible_dir in os.listdir(path_images):
				posible_dir = os.path.join(path_images, posible_dir)
				if os.path.isdir(posible_dir):
						for filename in os.listdir(posible_dir):
								images_paths.append(os.path.join(posible_dir, filename))
		return images_paths