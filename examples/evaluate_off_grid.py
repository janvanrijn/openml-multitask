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
    parser.add_argument('--data_file', type=str, default='../data/svm-offgrid.arff')
    parser.add_argument('--task_id_column', type=str, default='task_id')
    parser.add_argument('--hyperparameters', type=str, nargs='+', default=None)
    parser.add_argument('--y_column', type=str, default='y')
    parser.add_argument('--test_size', type=int, default=150)
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def format_data(args):
    with open(args.data_file, 'r') as fp:
        dataset = arff.load(fp)
    column_names = [att[0] for att in dataset['attributes']]
    # hyperparameter_columns_names = [att[0] for att in dataset['attributes']]
    # hyperparameter_columns_names.remove(args.task_id_column)
    # hyperparameter_columns_names.remove(args.y_column)
    if args.hyperparameters is None:
        legal_columns = [att[0] for att in dataset['attributes']]
    else:
        legal_columns = args.hyperparameters
        legal_columns.append(args.y_column)
        legal_columns.append(args.task_id_column)

    frame = pd.DataFrame(np.array(dataset['data']), columns=column_names)
    for column in frame:
        if column in legal_columns:
            try:
                frame[column] = frame[column].astype(float)
            except ValueError:
                pass
        else:
            del frame[column]
    frame = pd.get_dummies(frame)

    all_tasks = getattr(frame, args.task_id_column).unique()

    tasks_X_values = []
    tasks_y_values = []
    tasks = []
    for idx, task_id in enumerate(all_tasks):
        if args.num_tasks is not None and idx >= args.num_tasks:
            break
        tasks.append(int(task_id))
        current = frame.loc[frame[args.task_id_column] == task_id]
        del current[args.task_id_column]
        tasks_y_values.append(np.array(current[args.y_column]).reshape((-1, 1)))
        del current[args.y_column]
        tasks_X_values.append(current.as_matrix())
    return np.array(tasks_X_values, dtype=float), np.array(tasks_y_values, dtype=float), tasks


def run(args):
    np.random.seed(args.random_seed)
    tasks_X_values, tasks_y_values, tasks = format_data(args)
    num_tasks, num_obs, num_feats = tasks_X_values.shape
    # TODO: scale values per (2nd dimension)

    # make train and test sets
    test_indices = np.random.choice(num_obs, args.test_size, replace=False)
    train_indices= np.array(list(set(range(num_obs)) - set(test_indices)))

    task_X_train = tasks_X_values[:, train_indices, :]
    task_X_test = tasks_X_values[:, test_indices, :]
    task_y_train = tasks_y_values[:, train_indices, :]
    task_y_test = tasks_y_values[:, test_indices, :]

    models = [
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid()
    ]

    results = dict()
    for model in models:
        filename = 'offgrid.%s.%d.pkl' % (model.name, num_tasks)
        output_file = os.path.join(args.output_directory, filename)
        if os.path.isfile(output_file):
            print('Loaded %s from cache' %filename)
            with open(output_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
            continue
        print('Generating %s ' %filename)
        results[model.name] = dict()
        model.fit(task_X_train, task_y_train)

        for idx, task_id in enumerate(tasks):
            real_scores = task_y_test[idx].flatten()
            mean_prediction = model.predict(task_X_test, idx)
            spearman = scipy.stats.pearsonr(mean_prediction, real_scores)[0]
            mse = sklearn.metrics.mean_squared_error(real_scores, mean_prediction)
            results[model.name][task_id] = {'spearman': spearman, 'mse': mse}

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
        outputfile = os.path.join(parse_args().output_directory, 'offgrid-%s.pdf' %measure)
        multitask.plot.plot_boxplots(results, measure, outputfile)