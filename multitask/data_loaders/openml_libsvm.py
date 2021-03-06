import arff
import multitask
import numpy as np
import os
import pandas as pd


class OpenMLLibSVMDataLoader(object):

    name = 'OpenMLLibSVM'
    data_file = os.path.join(os.path.dirname(multitask.__file__), '../data/svm-offgrid.arff')

    @staticmethod
    def load_data(num_tasks=None, task_id_column='task_id', y_column='y', hyperparameters=None,
                  log_columns=['C', 'gamma', 'tol'], per_task_limit=None):
        with open(OpenMLLibSVMDataLoader.data_file, 'r') as fp:
            dataset = arff.load(fp)
        column_names = [att[0] for att in dataset['attributes']]
        # hyperparameter_columns_names = [att[0] for att in dataset['attributes']]
        # hyperparameter_columns_names.remove(task_id_column)
        # hyperparameter_columns_names.remove(y_column)
        if hyperparameters is None:
            legal_columns = [att[0] for att in dataset['attributes']]
        else:
            legal_columns = hyperparameters
            legal_columns.append(y_column)
            legal_columns.append(task_id_column)

        frame = pd.DataFrame(np.array(dataset['data']), columns=column_names)
        for column in frame:
            if column in legal_columns:
                try:
                    frame[column] = frame[column].astype(float)
                    # TODO: very important to represent log-scale parameters on the log scale
                    if column in log_columns:
                        frame[column] = np.log(frame[column])

                except ValueError:
                    pass
            else:
                del frame[column]
        frame = pd.get_dummies(frame)

        # make sure task idx is the last column
        all_columns = list(frame)
        all_columns.append(all_columns.pop(all_columns.index(task_id_column)))
        frame = frame.ix[:, all_columns]

        all_tasks_ids = getattr(frame, task_id_column).unique()
        task_indices = {task_id: idx for idx, task_id in enumerate(all_tasks_ids)}
        frame[task_id_column] = getattr(frame, task_id_column).map(task_indices)
        all_task_idx = getattr(frame, task_id_column).unique()
        if set(all_task_idx) != set(range(len(all_tasks_ids))):
            print(set(range(len(all_tasks_ids))))
            print(set(all_task_idx))
            raise ValueError('Something went wrong with renaming the task indices')

        tasks_X_values = []
        tasks_y_values = []

        # determine dataset wide bounds
        lower_bounds = np.array(frame.min(axis=0))[0:-2]  # remove task idx
        upper_bounds = np.array(frame.max(axis=0))[0:-2]  # remove task idx
        parameter_names = np.array(all_columns)[0:-2]

        for idx in all_task_idx:
            if num_tasks is not None and idx >= num_tasks:
                break
            current = frame.loc[frame[task_id_column] == idx]
            if per_task_limit is not None:
                current = current.head(per_task_limit)
            y_vals = np.array(current[y_column], dtype=float)

            tasks_y_values.append(y_vals)
            del current[y_column]
            tasks_X_values.append(current.as_matrix())
        return np.array(tasks_X_values, dtype=float), \
               np.array(tasks_y_values, dtype=float), \
               parameter_names, lower_bounds, upper_bounds
