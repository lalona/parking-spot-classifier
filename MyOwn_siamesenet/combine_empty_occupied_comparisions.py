import json

def main():
	# read
	dataset_compempty_path = 'dataset_cnrpark_labels_reduced_comparing-images_70.json'
	dataset_compoccupied_path = 'dataset_compoccupied_cnrpark_labels_reduced_comparing-images_70.json'
	with open(dataset_compempty_path) as f:
			dataset_compempty = json.load(f)

	with open(dataset_compoccupied_path) as f:
			dataset_compoccupied = json.load(f)

	dataset_comp = {}
	for (park_e, spaces_e), (park_c, spaces_c) in zip(dataset_compempty.items(), dataset_compoccupied.items()):
		dataset_comp[park_e] = {}
		for (space_e, comparissions_e), (space_c, comparissions_c) in zip(spaces_e.items(), spaces_c.items()):
			dataset_comp[park_e][space_e] = {'comparing_with_empty': comparissions_e[0]['comparing_with'], 'comparing_with_occupied': comparissions_c[0]['comparing_with'], 'comparissions': []}
			for comparission in comparissions_c:
					dataset_comp[park_e][space_e]['comparissions'].append({'comparing_to': comparission['comparing_to'], 'state': comparission['state']})

	with open('dataset_comp_cnrpark_labels_reduced_comparing-images_70.json', 'w') as outfile:
			json.dump(dataset_comp, outfile)


if __name__ == "__main__":
	main()