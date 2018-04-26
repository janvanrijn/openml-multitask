import arff
import argparse
import collections
import csv
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--input_directory', type=str, default='/home/janvanrijn/projects/TST/data/svm')
    parser.add_argument('--output_file', type=str, default='../data/svm-ongrid.arff')
    parser.add_argument('--xval_column_names', type=int, nargs='+',
                        default=['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'])
    parser.add_argument('--xval_column_idx', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--yval_column_idx', type=int, default=0)

    return parser.parse_args()


def run(args):
    if len(args.xval_column_idx) != len(args.xval_column_names):
        raise ValueError()

    # first enumerate over all datasets and fill config_dataset_result dict
    config_dataset_result = collections.defaultdict(dict)
    for file in os.listdir(args.input_directory):
        dataset_file = os.path.join(args.input_directory, file)
        with open(dataset_file, 'r') as fp:
            csvreader = csv.reader(fp, delimiter=' ')
            for row in csvreader:
                current_X = []
                current_y = float(row[args.yval_column_idx])
                for idx in args.xval_column_idx:
                    current_X.append(float(row[idx]))
                current_X = tuple(current_X)
                config_dataset_result[current_X][file] = current_y

    # assert that the grid is full
    distinct_tasks = set(os.listdir(args.input_directory))
    for config, dataset_result in config_dataset_result.items():
        if set(dataset_result.keys()) != distinct_tasks:
            raise ValueError('task set not ok for conig %s' %config)

    # create the arff
    attributes = []
    data = []

    # create arff attributes
    for idx, _ in enumerate(args.xval_column_idx):
        attribute = (args.xval_column_names[idx], 'NUMERIC')
        attributes.append(attribute)
    for filename in os.listdir(args.input_directory):
        attribute = ('y-on-' + filename, 'NUMERIC')
        attributes.append(attribute)

    # create arff data
    for config, dataset_result in config_dataset_result.items():
        current = list(config)
        for filename in os.listdir(args.input_directory):
            current.append(dataset_result[filename])
        data.append(np.array(current, dtype=float))
    data = np.array(data, dtype=float)

    num_configs = len(config_dataset_result)
    expected_shape = (num_configs, len(attributes))
    if data.shape != expected_shape:
        raise ValueError('data dimensions wrong. Expected %s; Got %s' %(expected_shape, data.shape))

    # create the arff object
    relation = 'ongrid-svm'
    description = 'Data obtained from Wistuba et. al; https://github.com/wistuba/TST'
    dataset = {'relation': relation, 'attributes': attributes, 'data': data, 'description': description}
    with open(args.output_file, 'w') as fp:
        fp.write(arff.dumps(dataset))


if __name__ == '__main__':
    run(parse_args())