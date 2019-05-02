
import json
from astropy.io import ascii
import argparse
from operator import itemgetter
import os
import csv


def getFileName(file_name):
		if len(file_name) >= 259:
				file_name = "\\\\?\\" + file_name
		return file_name

nets = 'malexnet,pnv9,pnv10'
def showTable(correct_images_by_parkinglot_details_, title, analyzed_file, title_orig=''):
		comparission_table = {'error': ['comp', 'por']}
		total_correct = 0
		total_failed = 0
		new_dirs = []
		comparissions_info = {}

		for key, values in correct_images_by_parkinglot_details_.items():
				if 'failed' not in values:
						values['error_por'] = 0
						values['failed'] = 0
				elif 'correct' not in values:
						values['error_por'] = 100
						values['correct'] = 0
				else:
					values['error_por'] = len(values['failed']) * 100 / (len(values['correct']) + len(values['failed']))
				values['key'] = key
				try:
						values['key_ord'] = int(key)
				except (ValueError, TypeError):
						values['key_ord'] = key

				new_dirs.append(values)

		new_dirs = sorted(new_dirs, key=itemgetter('key_ord'))
		for values in new_dirs:
			if 'failed' in values:
				correct_count = len(values['correct'])
				failed_count = len(values['failed'])
				comparission_table[values['key']] = ['{}/{}'.format(correct_count, failed_count)]
				comparission_table[values['key']].append('{:.2f}'.format(values['error_por']))
				total_correct += int(correct_count)
				total_failed += int(failed_count)

		comparission_table['total'] = ['{}/{}'.format(total_correct, total_failed),
									   '{}'.format(total_failed * 100 / (total_correct + total_failed))]
		comparissions_info[title] = comparission_table

		analyzed_dir, analyzed_filename = os.path.split(analyzed_file)
		nets_ = nets.split(',')
		net = ''
		for n in nets_:
			if n in analyzed_dir:
				net = n
				break

		print(analyzed_dir)
		if net == 'pnv1':
			net = 'pnv10'

		db_trained = ''
		if 'cnrpark' in analyzed_dir:
			if 'dataaug' in analyzed_dir:
				db_trained = 'cn_aug'
			else:
				db_trained = 'cn'
		else:
			if 'dataaug' in analyzed_dir:
				db_trained = 'pk_aug'
			else:
				db_trained = 'pk'

		analyzed_file_csv = analyzed_filename.replace('_tested_error_images_info', '')
		analyzed_file_csv = analyzed_file_csv.replace('_labels_reduced_comparing-images', '')
		analyzed_file_csv = analyzed_file_csv.replace('.json', '_{}_{}.csv'.format(net, db_trained))
		analyzed_file_csv = getFileName(os.path.join('analyzed_file', analyzed_file_csv))

		#print(analyzed_file_csv)
		for key, comp_info in comparissions_info.items():
				#print(comp_info)
				ascii.write(comp_info, 'temp.csv', format='csv',
										fast_writer=False)
				rows = []
				with open('temp.csv', 'r') as csvFile:
						reader = csv.reader(csvFile)
						for row in reader:
								rows.append(row)
				with open(analyzed_file_csv, 'a+') as fd:
						c = csv.writer(fd)
						c.writerow(['\n'])
						c.writerow([key])
						for row in rows:
								c.writerow(row)
				os.remove('temp.csv')

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-a", "--analyzed-file", type=str, required=True,
												help='Teh file with the details analized.')

		args = vars(parser.parse_args())

		analyzed_files_txt = args["analyzed_file"]

		with open(analyzed_files_txt, 'r') as f:
				content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		analyzed_files = [x.strip() for x in content]

		for a in analyzed_files:
				#print(a)
				analyzed_file = getFileName(a)
				with open(analyzed_file) as f:
						details = json.load(f)


				correct_images_by_parkinglot_space_details = details['by_space']
				correct_images_by_parkinglot_weather_details = details['by_date']
				correct_images_by_parkinglot_states_details = details['by_state']
				#print(correct_images_by_parkinglot_states_details)

				#correct_images_by_parkinglot_space_details
				#correct_images_by_parkinglot_weather_details

				showTable(correct_images_by_parkinglot_space_details, 'spaces', analyzed_file)
				showTable(correct_images_by_parkinglot_weather_details, 'date', analyzed_file)
				showTable(correct_images_by_parkinglot_states_details, 'state', analyzed_file)

if __name__ == "__main__":
		main()