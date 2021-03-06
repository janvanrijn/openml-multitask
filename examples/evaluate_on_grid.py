import argparse
import arff
import multitask
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats
import sklearn.metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--output_directory', type=str, default='/home/janvanrijn/experiments/multitask/multi/')
    parser.add_argument('--y_prefix', type=str, default='y-on-')
    parser.add_argument('--test_size', type=int, default=150)
    parser.add_argument('--num_tasks', type=int, default=25)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    return parser.parse_args()

def run(args):
    np.random.seed(args.random_seed)
    X_values, Y_values = multitask.data_loaders.WistubaLibSVMDataLoader.load_data_raw(num_tasks=args.num_tasks)
    num_obs, num_feats = X_values.shape
    _, num_tasks = Y_values.shape
    # TODO: scale values?

    # make train and test sets
    test_indices = np.random.choice(num_obs, args.test_size, replace=False)
    train_indices= np.array(list(set(range(num_obs)) - set(test_indices)))

    X_train = X_values[train_indices, :]
    X_test = X_values[test_indices, :]
    Y_train = Y_values[train_indices, :]
    Y_test = Y_values[test_indices, :]

    models = [
        multitask.models_ongrid.MetaMultiOutputGPOngrid(),
        multitask.models_ongrid.MetaRandomForestOngrid(),
        multitask.models_ongrid.MetaSingleOutputGPOngrid()
    ]

    results = dict()
    for model in models:
        filename = 'ongrid.%s.%d.pkl' % (model.name, num_tasks)
        output_file = os.path.join(args.output_directory, filename)
        if os.path.isfile(output_file):
            print('Loaded %s from cache' %filename)
            with open(output_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
            continue
        print('Generating %s ' %filename)
        results[model.name] = dict()
        model.fit(X_train, Y_train)

        for idx in range(num_tasks):
            real_scores = Y_test[:, idx]
            mean_prediction = model.predict(X_test, idx)
            spearman = scipy.stats.pearsonr(mean_prediction, real_scores)[0]
            mse = sklearn.metrics.mean_squared_error(real_scores, mean_prediction)
            results[model.name][idx] = {'spearman': spearman, 'mse': mse}

        try:
            os.makedirs(args.output_directory)
        except FileExistsError:
            pass
        with open(output_file, 'wb') as fp:
            pickle.dump(results[model.name], fp)
    return results


if __name__ == '__main__':
    results = run(parse_args())

    for measure in ['spearman', 'mse']:
        outputfile = os.path.join(parse_args().output_directory, 'ongrid-%s.%s' %(measure, parse_args().extension))
        multitask.plot.plot_boxplots(results, measure, measure + ' on grid', outputfile)
