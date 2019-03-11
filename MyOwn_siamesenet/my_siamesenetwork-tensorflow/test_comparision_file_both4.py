import argparse
import json
from operator import itemgetter
import os
import msvcrt as m

def getAproximation_old(c, min_=0.0, max_=1.0):
		min_approximation = c['pred_similarity'] - min_
		max_approximation = max_ - c['pred_similarity']
		if min_approximation < max_approximation:
				return max_approximation
		else:
				print('something')
				return min_approximation


def getAproximation(c, max_=1.0):
		return c['pred_similarity']



def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-c", "--comparission-dir", type=str, required=True,
												help='Path to comparission directory.')
		parser.add_argument("-d", "--database", type=str, required=True,
												help='Database.')

		args = vars(parser.parse_args())

		comparission_dir = args["comparission_dir"]
		database = args["database"]

		if database == 'cnrpark':
				comparission_occupied_file = os.path.join(comparission_dir,
																									'comparision_coccupied_images_info_dataset_compoccupied_cnrpark_labels_reduced_comparing-images_50.txt')
				comparission_empty_file = os.path.join(comparission_dir,
																							 'comparision_images_info_dataset_cnrpark_labels_reduced_comparing-images_50.txt')
		elif database == 'pklot':
			comparission_occupied_file = os.path.join(comparission_dir, 'comparision_coccupied_images_info_dataset_compoccupied_pklot_labels_reduced_comparing-images_50.txt')
			comparission_empty_file = os.path.join(comparission_dir,
																							'comparision_images_info_dataset_pklot_labels_reduced_comparing-images_50.txt')
		else:
				print('This {} database doesnt exist.'.format(database))
				return

		with open(comparission_occupied_file) as f:
				occupied_comparission_values = json.load(f)
		with open(comparission_empty_file) as f:
				empty_comparission_values = json.load(f)

		total_count = 0
		total_failed = 0
		errors = []
		for (occupied_comparing_with, occupied_comparissions_results), (empty_comparing_with, empty_comparissions_results) in zip(
						occupied_comparission_values.items(), empty_comparission_values.items()):
				count = 0
				failed = 0
				for occupied_c, empty_c in zip(occupied_comparissions_results, empty_comparissions_results):
						#print('State: {}'.format(occupied_c['similarity']))
						#print('Occupied')
						occupied_pred_value = getAproximation(c=occupied_c)
						#print('Empty')
						empty_pred_value = getAproximation(c=empty_c)

						count += 1
						if occupied_pred_value < empty_pred_value and occupied_c['similarity'] == 0:
								failed += 1
						if occupied_pred_value > empty_pred_value and occupied_c['similarity'] == 1:
								failed += 1
						#m.getch()
				errors.append(failed * 100 / count)

				total_count += count
				total_failed += failed

		#for i, e in enumerate(sorted(errors)):
		#		print('{}  {}'.format(i, e))
		print('Total err same: {} {} {}'.format(total_failed * 100 / total_count, total_count, total_failed))




if __name__ == "__main__":
		main()



