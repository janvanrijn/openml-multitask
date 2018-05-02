import arff
import numpy as np


class WistubaLibSVMDataLoader(object):

    name = 'WistubaLibSVM'

    @staticmethod
    def load_data_raw(data_file='../data/svm-ongrid.arff', num_tasks=None, y_prefix='y-on-',
                    x_column_names=['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree']):
        with open(data_file, 'r') as fp:
            dataset = arff.load(fp)
        x_indices = list()
        y_indices = list()
        for idx, (column, type) in enumerate(dataset['attributes']):
            if column in x_column_names:
                x_indices.append(idx)
            elif num_tasks is None or len(y_indices) < num_tasks:
                if column.startswith(y_prefix):
                    y_indices.append(idx)

        if len(x_indices) != len(x_column_names):
            raise ValueError('Couldn\'t find all hyperparameter columns: ')

        data = np.array(dataset['data'])
        return np.array(data[:, x_indices], dtype=float), np.array(data[:, y_indices], dtype=float)

    @staticmethod
    def load_data(num_tasks=None):
        raw_X_data, raw_Y_data = WistubaLibSVMDataLoader.load_data_raw(num_tasks=num_tasks)
        num_obs, num_feats = raw_X_data.shape
        _, num_tasks = raw_Y_data.shape

        tasks_X_data = []
        tasks_y_data = []
        for idx in range(num_tasks):
            X_data = np.zeros((num_obs, num_feats + 1))
            X_data[:, :-1] = raw_X_data
            X_data[:, -1] = idx
            tasks_X_data.append(X_data)
            tasks_y_data.append(raw_Y_data[:, idx])

        return np.array(tasks_X_data, dtype=float), np.array(tasks_y_data, dtype=float)
