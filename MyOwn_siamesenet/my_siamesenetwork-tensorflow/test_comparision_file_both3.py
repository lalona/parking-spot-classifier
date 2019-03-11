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
				#if c['pred_similarity'] > 0.0:
				if c['comparing_to'] == comparing_with:
						min_ = c['pred_similarity']
						#print('yeah {}'.format(min_))
						break
		max_ = comparissions_results[-1]['pred_similarity']
		#return min_, max_
		return 0.0, max_


def checkPrediction(c, min_, max_, min_sim, max_sim):
		min_approximation = c['pred_similarity'] - min_
		max_approximation = max_ - c['pred_similarity']

		if c['similarity'] == max_sim:
				if min_approximation < max_approximation:
						return (False, max_approximation)
				else:
						return (True, min_approximation)
		#return (True, min_approximation)
		if c['similarity'] == min_sim:
				if min_approximation > max_approximation:
						return (False, min_approximation)
				else:
						return (True, max_approximation)
		#return (True, min_approximation)

def getState(c, min_, max_, comparing_with_state):
		min_approximation = c['pred_similarity'] - min_
		max_approximation = max_ - c['pred_similarity']

		if comparing_with_state == 'occupied':
			if min_approximation < max_approximation:
					return ('occupied', max_approximation)
			else:
					return ('empty', min_approximation)
		elif comparing_with_state == 'empty':
			if min_approximation < max_approximation:
					return ('empty', max_approximation)
			else:
					return ('occupied', min_approximation)

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-c", "--comparission-dir", type=str, required=True,
												help='Path to comparission directory.')

		args = vars(parser.parse_args())

		comparission_dir = args["comparission_dir"]

		comparission_occupied_file = os.path.join(comparission_dir, 'comparision_coccupied_images_info_dataset_compoccupied_pklot_labels_reduced_comparing-images_50.txt')
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
		count_pred = 0
		failed_by = {'0': {'count': 0, 'failed': 0}, '1': {'count': 0, 'failed': 0}, '2': {'count': 1, 'failed': 0}}
		for (occupied_comparing_with, occupied_comparissions_results), (empty_comparing_with, empty_comparissions_results) in zip(
						occupied_comparission_values.items(), empty_comparission_values.items()):


				occupied_min, occupied_max = getMinMax(occupied_comparissions_results, occupied_comparing_with)
				empty_min, empty_max = getMinMax(empty_comparissions_results, empty_comparing_with)
				count = 0
				failed_empty = 0
				failed_occupied = 0
				failed_same = 0

				for occupied_c, empty_c in zip(occupied_comparissions_results, empty_comparissions_results):

						occupied_pred_state, occupied_pred_aprox = getState(occupied_c, min_= occupied_min, max_=occupied_max, comparing_with_state='occupied')
						empty_pred_state, empty_pred_aprox = getState(empty_c, min_=empty_min, max_=empty_max, comparing_with_state='empty')
						count += 1
						correct_answer = 'empty' if occupied_c['similarity'] == 0 else 'occupied'

						pred = ''
						# si al compararlo con ocupado dice que es diferente osea vacio, entonces confio en esa prediccion
						# lo mismo para el vacio cuando predice que esta ocupado
						_by = ''
						if occupied_pred_state == empty_pred_state:
								pred = occupied_pred_state
								_by = '0'
						else:
								pred = 'occupied'
								_by = '1'


						# si no se ninguno de los dos if de arriba

						if len(pred) == 0:
								count_pred += 1
								#print(pred)


						if correct_answer != empty_pred_state:
								failed_same += 1
								failed_by[_by]['failed'] += 1
						failed_by[_by]['count'] += 1







				total_count += count
				total_failed_empty += failed_empty
				total_failed_occupied += failed_occupied
				total_failed_same += failed_same

				#print('err: {} {} {}'.format(failed * 100 / count, count, failed))

		print('Total err empty: {} {} {}'.format(total_failed_empty * 100 / total_count, total_count, total_failed_empty))

		print('Total err occupied: {} {} {}'.format(total_failed_occupied * 100 / total_count, total_count, total_failed_occupied))

		print('Total err same: {} {} {}'.format(total_failed_same * 100 / total_count, total_count, total_failed_same))

		for key, values in failed_by.items():
			print('Total err same: {} {} {}'.format(values['failed'] * 100 / values['count'], values['count'], values['failed']))

		print(count_pred)

if __name__ == "__main__":
		main()



