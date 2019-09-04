import argparse
import csv
import logging
import numpy as np
import os
import pandas as pd


# use either of following github repo:
# - https://github.com/wistuba/TST/
# - https://github.com/janvanrijn/TST (fork)
def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--input_directory', type=str, default=os.path.expanduser('~/projects/TST/data/svm'))
    parser.add_argument('--output_file_dir', type=str, default=os.path.expanduser('~/experiments/openml-multitask'))
    parser.add_argument('--output_file_name', type=str, default='svm-ongrid')
    parser.add_argument('--xval_column_names', type=str, nargs='+',
                        default=['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'])
    parser.add_argument('--xval_column_prefix', type=str, default='perf-on-')
    parser.add_argument('--xval_column_idx', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--yval_column_idx', type=int, default=0)

    return parser.parse_args()


def run(args):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    if len(args.xval_column_idx) != len(args.xval_column_names):
        raise ValueError()

    # all datasets that we can expect
    dataset_files = os.listdir(args.input_directory)

    df = pd.DataFrame(columns=args.xval_column_names + ['%s%s' % (args.xval_column_prefix, dataset)
                                                                  for dataset in dataset_files])
    df = df.set_index(args.xval_column_names)

    for f_idx, file in enumerate(dataset_files):
        dataset_file = os.path.join(args.input_directory, file)
        column = '%s%s' % (args.xval_column_prefix, file)
        with open(dataset_file, 'r') as fp:
            csvreader = csv.reader(fp, delimiter=' ')
            for row in csvreader:
                current_X = []
                current_y = float(row[args.yval_column_idx])
                for idx in args.xval_column_idx:
                    current_X.append(float(row[idx]))
                num_nans = sum(np.isnan(np.array(current_X)))
                if num_nans > 0:
                    raise ValueError()
                if f_idx == 0:
                    df.loc[tuple(current_X)] = np.nan
                df.loc[tuple(current_X)][column] = current_y

    if df.isnull().sum().sum() > 0:
        raise ValueError()

    os.makedirs(args.output_file_dir, exist_ok=True)
    output_file = os.path.join(args.output_file_dir, '%s.csv' % args.output_file_name)
    df.to_csv(output_file)
    logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
