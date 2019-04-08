
import json
from astropy.table import Table, Column
from astropy.io import ascii
import argparse
from operator import itemgetter
import os
import csv


def getFileName(file_name):
		if len(file_name) >= 259:
				file_name = "\\\\?\\" + file_name
		return file_name


def showTable(correct_images_by_parkinglot_details_, title, analyzed_file, title_orig=''):
		comparission_table = {'error': ['comp', 'por']}
		total_correct = 0
		total_failed = 0
		show = True
		new_dirs = []
		for key, values in correct_images_by_parkinglot_details_.items():
				if 'failed' in values or 'correct' in values:
						if 'failed' not in values:
								values['error_por'] = 0
								values['failed'] = 0
						elif 'correct' not in values:
								values['error_por'] = 100
								values['correct'] = 0
						else:
							values['error_por'] = values['failed'] * 100 / values['correct']
						values['key'] = key
						new_dirs.append(values)
				else:
						showTable(values, key, analyzed_file, title)
						show = False

		if show:
				new_dirs = sorted(new_dirs, key=itemgetter('error_por'))
				for values in new_dirs:
						if 'failed' in values:
								comparission_table[values['key']] = ['{}/{}'.format(values['correct'], values['failed'])]
								comparission_table[values['key']].append('{:.2f}'.format(values['error_por']))
								total_correct += int(values['correct'])
								total_failed += int(values['failed'])

				comparission_table['total'] = ['{}/{}'.format(total_correct, total_failed),
																			 '{}'.format(total_failed * 100 / total_correct)]

				#print(comparission_table)
				print(title)
				#print(Table(comparission_table))

				ascii.write(comparission_table, 'analyzed_{}_{}_temp.csv'.format(title, title_orig), format='csv', fast_writer=False)
				rows = []
				with open('analyzed_{}_{}_temp.csv'.format(title, title_orig), 'r') as csvFile:
						reader = csv.reader(csvFile)
						for row in reader:
								rows.append(row)
				if os.path.isfile('analyzed_{}_{}.csv'.format(title, title_orig)):
						with open('analyzed_{}_{}.csv'.format(title, title_orig), 'r') as csvFile:
								reader = csv.reader(csvFile)
								for row in reader:
										if len(row) > 0:
												if row[0] == 'file':
														if row[1] == analyzed_file:
																print('This one was already made')
																return
				with open('analyzed_{}_{}.csv'.format(title, title_orig), 'a+') as fd:
						c = csv.writer(fd)
						c.writerow(['\n'])
						c.writerow(['file', analyzed_file])
						for row in rows:
								c.writerow(row)
				os.remove('analyzed_{}_{}_temp.csv'.format(title, title_orig).format(title))




def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-a", "--analyzed-file", type=str, required=True,
												help='Teh file with the details analized.')

		args = vars(parser.parse_args())

		analyzed_file = args["analyzed_file"]

		analyzed_dir, analyzed_filename = os.path.split(analyzed_file)

		if 'pklot' in analyzed_file:
				database = 'pklot'
		else:
				database = 'cnrpark'

		with open(getFileName(analyzed_file)) as f:
				details = json.load(f)

		correct_images_by_parkinglot_details = details['correct_images_by_parkinglot_details']
		correct_images_by_parkinglot_space_details = details['correct_images_by_parkinglot_space_details']
		correct_images_by_parkinglot_weather_details = details['correct_images_by_parkinglot_weather_details']

		#correct_images_by_parkinglot_space_details
		#correct_images_by_parkinglot_weather_details

		showTable(correct_images_by_parkinglot_space_details, 'spaces', analyzed_file)
		showTable(correct_images_by_parkinglot_weather_details, 'weather', analyzed_file)

if __name__ == "__main__":
		main()