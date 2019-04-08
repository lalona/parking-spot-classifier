import argparse
import json
from operator import itemgetter
import os

def getDateFromPath(path):
	return path.split('\\')[8]

def getMinMax(comparissions_results, comparing_with):
		comparissions_results = sorted(comparissions_results, key=itemgetter('pred_similarity'))
		min_ = 0.0
		for c in comparissions_results:
				if c['comparing_to'] == comparing_with:
						#print('yeah')
						min_ = c['pred_similarity']
						break
		max_ = comparissions_results[-1]['pred_similarity']
		return 0.0, 1.0


def checkPrediction(c, min_, max_, min_sim, max_sim):
		min_approximation = c['pred_similarity'] - min_
		max_approximation = max_ - c['pred_similarity']
		if c['similarity'] == max_sim:
				if min_approximation < max_approximation:
						return (False, max_approximation)
				else:
						return (True, min_approximation)
		if c['similarity'] == min_sim:
				if min_approximation > max_approximation:
						return (False, min_approximation)
				else:
						return (True, max_approximation)


def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-c", "--comparission-dir", type=str, required=True,
												help='Path to comparission directory.')

		args = vars(parser.parse_args())

		comparission_dir = args["comparission_dir"]

		#comparission_occupied_file = os.path.join(comparission_dir, 'comparision_coccupied_images_info_dataset_compoccupied_pklot_labels_reduced_comparing-images_50.txt')
		comparission_occupied_file = os.path.join(comparission_dir,
																							'comparision_images_info_dataset_compoccupied_pklot_labels_reduced_comparing-images_50.txt')
		comparission_empty_file = os.path.join(comparission_dir,
																							'comparision_images_info_dataset_pklot_labels_reduced_comparing-images_50.txt')

		with open(comparission_occupied_file) as f:
				occupied_comparission_values = json.load(f)
		with open(comparission_empty_file) as f:
				empty_comparission_values = json.load(f)

		total_count = 0
		total_failed_empty = 0
		total_failed_occupied = 0
		total_failed_same = 0

		empty_total = 0.0
		occupied_total = 0.0
		total_failed_t = 0
		e_count = 0
		o_count = 0
		for (occupied_comparing_with, occupied_comparissions_results), (empty_comparing_with, empty_comparissions_results) in zip(
						occupied_comparission_values.items(), empty_comparission_values.items()):


				occupied_min, occupied_max = getMinMax(occupied_comparissions_results, occupied_comparing_with)
				empty_min, empty_max = getMinMax(empty_comparissions_results, empty_comparing_with)

				#print('{} {}  {} {}'.format(occupied_min, occupied_max, empty_min, empty_max))
				count = 0
				failed_empty = 0
				failed_occupied = 0
				failed_same = 0

				for occupied_c, empty_c in zip(occupied_comparissions_results, empty_comparissions_results):

						occupied_prediction_check, occupied_pred_value = checkPrediction(c=occupied_c, min_=occupied_min, max_=occupied_max, min_sim=1, max_sim=0)
						empty_prediction_check, empty_pred_value = checkPrediction(c=empty_c, min_=empty_min, max_=empty_max, min_sim=0,
																												max_sim=1)

						count += 1
						if not occupied_prediction_check:
								failed_occupied += 1
						if not empty_prediction_check:
								failed_empty += 1
						#print('{} {}'.format(occupied_pred_value, empty_pred_value))
						if occupied_pred_value >= empty_pred_value and occupied_c['similarity'] == 0:
								failed_same += 1
						if occupied_pred_value <= empty_pred_value and occupied_c['similarity'] == 1:
								failed_same += 1

						#if not occupied_prediction_check and not empty_prediction_check:
						#		failed_same += 1

				total_count += count
				total_failed_empty += failed_empty
				total_failed_occupied += failed_occupied
				total_failed_same += failed_same
				if (failed_same * 100 / count) > 10.0:
						print('Total err same: {} {} {}'.format(failed_same * 100 / count, count,
																										failed_same))


				#print('err: {} {} {}'.format(failed * 100 / count, count, failed))

		print('Total err empty: {} {} {}'.format(total_failed_empty * 100 / total_count, total_count, total_failed_empty))

		print('Total err occupied: {} {} {}'.format(total_failed_occupied * 100 / total_count, total_count, total_failed_occupied))

		print('Total err same: {} {} {}'.format(total_failed_same * 100 / total_count, total_count, total_failed_same))




if __name__ == "__main__":
		main()



