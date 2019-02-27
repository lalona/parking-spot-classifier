import argparse
import json
from operator import itemgetter

def getDateFromPath(path):
	return path.split('\\')[8]

def main():
		parser = argparse.ArgumentParser(description='Select the type of reduced.')
		parser.add_argument("-c", "--comparission-file", type=str, required=True,
												help='Path to comparission file.')

		args = vars(parser.parse_args())

		comparission_file = args["comparission_file"]

		with open(comparission_file) as f:
				comparission_values = json.load(f)

		total_count = 0
		total_failed = 0
		empty_total = 0.0
		occupied_total = 0.0
		total_failed_t = 0
		e_count = 0
		o_count = 0
		for comparing_with, comparissions_results in comparission_values.items():
				comparissions_results = sorted(comparissions_results, key=itemgetter('pred_similarity'))
				min_ = 0.0
				for c in comparissions_results:
						if c['pred_similarity'] > 0.0:
								min_ = c['pred_similarity']
								break
				max_ = comparissions_results[-1]['pred_similarity']

				count = 0
				failed = 0
				count_t = 0
				failed_t = 0
				for c in comparissions_results:
						min_approximation = c['pred_similarity'] - min_
						max_approximation = max_ -  c['pred_similarity']
						if min_approximation < max_approximation and c['similarity'] == 0:
								failed += 1
						if min_approximation > max_approximation and c['similarity'] == 1:
								failed += 1

						treshold = 0.02
						if c['pred_similarity'] < treshold and c['similarity'] == 0:
								failed_t += 1
						if c['pred_similarity'] >= treshold and c['similarity'] == 1:
								failed_t += 1

						if c['similarity'] == 1:
								empty_total += c['pred_similarity']
								e_count += 1
						else:
								occupied_total += c['pred_similarity']
								o_count += 1

						count += 1
				total_count += count
				total_failed += failed
				total_failed_t += failed_t


				print('err: {} {} {}'.format(failed * 100 / count, count, failed))

		print('Total err: {} {} {}'.format(total_failed * 100 / total_count, total_count, total_failed))

		print('Total err: {} {} {}'.format(total_failed_t * 100 / total_count, total_count, total_failed_t))

		print('Total err: {} {} {}'.format(empty_total / e_count, occupied_total / o_count, total_failed))




if __name__ == "__main__":
		main()



