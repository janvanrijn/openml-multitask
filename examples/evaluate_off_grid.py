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
    parser.add_argument('--test_size', type=int, default=150)
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--use_cache', action='store_true', default=False)
    return parser.parse_args()


def run(args):
    np.random.seed(args.random_seed)
    tasks_X_values, tasks_y_values = multitask.data_loaders.OpenMLLibSVMDataLoader.load_data(num_tasks=args.num_tasks)
    num_tasks, num_obs, num_feats = tasks_X_values.shape

    # make train and test sets
    test_indices = np.random.choice(num_obs, args.test_size, replace=False)
    train_indices= np.array(list(set(range(num_obs)) - set(test_indices)))

    task_X_train = tasks_X_values[:, train_indices, :]
    task_X_test = tasks_X_values[:, test_indices, :]
    task_y_train = tasks_y_values[:, train_indices]
    task_y_test = tasks_y_values[:, test_indices]

    models = [
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedRFOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid()
    ]

    results = dict()
    for model in models:
        filename = 'offgrid.%s.%d.pkl' % (model.name, num_tasks)
        output_file = os.path.join(args.output_directory, filename)
        if os.path.isfile(output_file) and args.use_cache:
            print(multitask.utils.get_time(), 'Loaded %s from cache' %filename)
            with open(output_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
            continue
        print(multitask.utils.get_time(), 'Generating %s ' %filename)
        results[model.name] = dict()
        model.fit(task_X_train, task_y_train)

        for idx in range(num_tasks):
            real_scores = task_y_test[idx].flatten()
            mean_prediction = model.predict(task_X_test, idx)
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
        outputfile = os.path.join(parse_args().output_directory, 'offgrid-%d-%s.%s' %(parse_args().num_tasks, measure, parse_args().extension))
        multitask.plot.plot_boxplots(results, measure, measure + ' off grid', outputfile)
