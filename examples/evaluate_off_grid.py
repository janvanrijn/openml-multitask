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
    parser.add_argument('--log_columns', type=str, nargs='+', default=['C', 'gamma', 'tol'])
    parser.add_argument('--y_column', type=str, default='y')
    parser.add_argument('--test_size', type=int, default=150)
    parser.add_argument('--num_tasks', type=int, default=25)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--use_cache', action='store_true', default=False)
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
                # TODO: very important to model logscale parameters on the log scale
                if column in args.log_columns:
                    frame[column] = np.log(frame[column])

            except ValueError:
                pass
        else:
            del frame[column]
    frame = pd.get_dummies(frame)

    # make sure task idx is the last column
    all_columns = list(frame)
    all_columns.append(all_columns.pop(all_columns.index(args.task_id_column)))
    frame = frame.ix[:, all_columns]

    all_tasks_ids = getattr(frame, args.task_id_column).unique()
    task_indices = {task_id: idx for idx, task_id in enumerate(all_tasks_ids)}
    frame[args.task_id_column] = getattr(frame, args.task_id_column).map(task_indices)
    all_task_idx = getattr(frame, args.task_id_column).unique()
    if set(all_task_idx) != set(range(len(all_tasks_ids))):
        print(set(range(len(all_tasks_ids))))
        print(set(all_task_idx))
        raise ValueError('Something went wrong with renaming the task indices')

    tasks_X_values = []
    tasks_y_values = []

    for idx in all_task_idx:
        if args.num_tasks is not None and idx >= args.num_tasks:
            break
        current = frame.loc[frame[args.task_id_column] == idx]
        y_vals = np.array(current[args.y_column], dtype=float)

        tasks_y_values.append(y_vals)
        del current[args.y_column]
        tasks_X_values.append(current.as_matrix())
    return np.array(tasks_X_values, dtype=float), np.array(tasks_y_values, dtype=float)


def run(args):
    np.random.seed(args.random_seed)
    tasks_X_values, tasks_y_values = format_data(args)
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
            print('Loaded %s from cache' %filename)
            with open(output_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
            continue
        print('Generating %s ' %filename)
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
        outputfile = os.path.join(parse_args().output_directory, 'offgrid-%s.%s' %(measure, parse_args().extension))
        multitask.plot.plot_boxplots(results, measure, measure + ' off grid', outputfile)
