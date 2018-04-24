import argparse
import arff
import collections
import GPy
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--plot_directory', type=str, default='/home/janvanrijn/experiments/multitask/multi/')
    parser.add_argument('--data_file', type=str, default='../data/openml-svm.arff')
    parser.add_argument('--task_id_column', type=str, default='task_id')
    parser.add_argument('--y_column', type=str, default='y')
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def format_data(args):
    with open(args.data_file, 'r') as fp:
        dataset = arff.load(fp)
    column_names = [att[0] for att in dataset['attributes']]
    hyperparameter_columns_names = [att[0] for att in dataset['attributes']]
    hyperparameter_columns_names.remove(args.task_id_column)
    hyperparameter_columns_names.remove(args.y_column)

    frame = pd.DataFrame(np.array(dataset['data']), columns=column_names)
    for column in frame:
        try:
            frame[column] = frame[column].astype(float)
        except ValueError:
            pass
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
        print(current.columns.values)
        tasks_y_values.append(current[args.y_column])
        del current[args.y_column]
        tasks_X_values.append(current.as_matrix())

    return np.array(tasks_X_values, dtype=float), np.array(tasks_y_values, dtype=float), tasks


def run(args):
    np.random.seed(args.random_seed)
    tasks_X_values, tasks_y_values, tasks = format_data(args)
    print(tasks_X_values.shape)
    num_tasks, num_obs, num_feats = tasks_X_values.shape

    # make train and test sets
    test_indices = np.random.choice(num_obs, args.test_size, replace=False)
    train_indices= np.array(list(set(range(num_obs)) - set(test_indices)))

    task_X_train = tasks_X_values[:,train_indices ,:]
    task_X_test = tasks_X_values[:,test_indices ,:]
    task_y_train = tasks_y_values[:,train_indices]
    task_y_test = tasks_y_values[:,test_indices]

    # train the model
    kernel = GPy.kern.Matern32(1)
    icm = GPy.util.multioutput.ICM(input_dim=num_feats, num_outputs=num_tasks, kernel=kernel)
    m = GPy.models.GPCoregionalizedRegression(task_X_train, task_y_train, kernel=icm)
    # For this kernel, B.kappa encodes the variance now.
    m['.*Mat32.var'].constrain_fixed(1.)
    m.optimize()


if __name__ == '__main__':
    run(parse_args())
