import arff
import numpy as np
import pandas as pd


class WistubaLibSVMDataLoader(object):

    name = 'WistubaLibSVM'

    @staticmethod
    def _datafile_to_dataframe(data_file):
        with open(data_file, 'r') as fp:
            arff_dataset = arff.load(fp)
        frame = pd.DataFrame(data=arff_dataset['data'],
                             columns=[name for name, datatype in arff_dataset['attributes']],
                             dtype=float)
        return frame

    @staticmethod
    def _stack_per_task_data(raw_X_data, raw_Y_data):
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

    @staticmethod
    def load_data_raw(data_file='../data/svm-ongrid.arff', num_tasks=None, y_prefix='y-on-',
                      x_column_names=['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'],
                      filter_fn=None):
        frame = WistubaLibSVMDataLoader._datafile_to_dataframe(data_file)
        if filter_fn is not None:
            frame = filter_fn(frame)

        x_indices = [idx for idx, col in enumerate(frame.columns) if col in x_column_names]
        y_indices = [idx for idx, col in enumerate(frame.columns) if col.startswith(y_prefix)]
        if num_tasks is not None:
            if len(y_indices) < num_tasks:
                raise ValueError('Not enough tasks .. ')
            y_indices = y_indices[0:num_tasks]

        if len(x_indices) != len(x_column_names):
            raise ValueError('Could not find all hyperparameter columns, expected %d got %d' % (len(x_column_names),
                                                                                                len(x_indices)))

        return frame.as_matrix()[:, x_indices], frame.as_matrix()[:, y_indices]

    @staticmethod
    def load_data(num_tasks=None):
        raw_X_data, raw_Y_data = WistubaLibSVMDataLoader.load_data_raw(num_tasks=num_tasks)
        return WistubaLibSVMDataLoader._stack_per_task_data(raw_X_data, raw_Y_data)

    @staticmethod
    def load_data_rbf_fixed_complexity():
        def filter_fn(df):
            df = df.sort_values(by=['gamma'], axis=0)
            df = df.loc[(df['kernel_rbf'] == float(1)) & (df['c'] == float(0))]
            return df

        raw_X_data, raw_Y_data = WistubaLibSVMDataLoader.load_data_raw(x_column_names=['gamma'],
                                                                       filter_fn=filter_fn)

        assert raw_X_data.shape == (14, 1)   # manually checked, adapt if needed
        assert raw_Y_data.shape == (14, 50)  # manually checked, adapt if needed
        return WistubaLibSVMDataLoader._stack_per_task_data(raw_X_data, raw_Y_data)
