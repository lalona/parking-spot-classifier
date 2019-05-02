import argparse
import json
import os
from operator import itemgetter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Model used for the accuracy test")
    args = vars(ap.parse_args())
    model_name = args['model']

    with open('accuracy_test_file.json', 'r') as json_file:
        accuracy_tests = json.load(json_file)

    accuracy_tests_model = accuracy_tests[model_name]

    map = []
    for space_id, accuracy_test in accuracy_tests_model.items():
        accuracy_tests = sorted(accuracy_test, key=itemgetter('err'))
        best_accuracy_test = accuracy_tests[0]
        best_accuracy_test.pop('err_by_day', None)
        print(best_accuracy_test)
        best_accuracy_test.pop('err', None)

        best_accuracy_test['id'] = space_id

        map.append(best_accuracy_test)

    model_name = os.path.splitext(model_name)[0]
    map_name = 'map_{}.json'.format(model_name)

    with open(map_name, 'w') as file:
        json.dump(map, file)



if __name__ == "__main__":
    main()