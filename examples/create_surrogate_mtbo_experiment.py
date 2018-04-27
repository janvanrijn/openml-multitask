import argparse
import csv
import numpy as np
import os
import pickle
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--directory', type=str, default='../data/svm_on_mnist_grid')
    parser.add_argument('--n_estimators', type=int, default=128)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--log', action='store_true', default=True)

    return parser.parse_args()


def run(args):
    configs_file = os.path.join(args.directory, 'configs.csv')

    vals_X = list()

    with open(configs_file, 'r') as fp:
        for row in csv.reader(fp):
            vals_X.append(row)
    vals_X = np.array(vals_X, dtype=float)

    response_values = [('costs.csv', 'rf_cost_surrogate_svm.pkl'), ('fvals.csv', 'rf_surrogate_svm.pkl')]
    for input_file, output_file in response_values:
        input_path = os.path.join(args.directory, input_file)
        output_path = os.path.join(args.directory, output_file)
        vals = list()
        with open(input_path, 'r') as fp:
            for row in csv.reader(fp):
                current = float(row[0])
                if args.log is True:
                    current = np.log(current)
                vals.append(current)

        vals = np.array(vals, dtype=float)
        surrogate = sklearn.ensemble.RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state)
        surrogate.fit(vals_X, vals)
        with open(output_path, 'wb') as fp:
            pickle.dump(surrogate, fp)


if __name__ == '__main__':
    run(parse_args())
