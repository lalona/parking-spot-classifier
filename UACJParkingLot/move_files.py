import shutil
import os
from tqdm import tqdm
#dest1 = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ\\30-11-2018'
dest1 = 'C:\\Eduardo\\tesis\\datasets\\uacj\\30-11-2018'

#directory = 'C:\\Eduardo\\ProyectoFinal\\Pruebas_UACJ'
directory = 'C:\\Eduardo\\tesis\\datasets\\uacj'
def main():
	subdirectories = [x[0] for x in os.walk(directory)]
	subdirectories_ = []

	if not os.path.isdir(dest1):
		os.mkdir(dest1)

	for s in subdirectories:
			print(s)
			folder = os.path.basename(os.path.normpath(s))
			print(s)
			if folder.startswith('30_'):
				subdirectories_.append(s)

	for s in tqdm(subdirectories_):
		files = os.listdir(s)
		for f in files:
			shutil.move(os.path.join(s, f), dest1)

if __name__ == "__main__":
    main()
