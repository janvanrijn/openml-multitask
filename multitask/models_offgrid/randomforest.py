import multitask.utils
import numpy as np
import sklearn.ensemble


class MetaRandomForestOffgrid(object):

    def __init__(self):
        self.models = dict()
        self.name = 'RandomForest'

    def get_name(self, num_tasks, num_obs):
        return self.name

    @multitask.utils.fit_and_measure_time
    def fit(self, X_train, y_train):
        num_obs, num_feats = X_train.shape
        assert y_train.shape == (num_obs, 1)

        all_tasks = np.unique(X_train[:, -1])
        for task_idx in list(all_tasks):
            indices = X_train[:, -1] == task_idx
            task_X_train = X_train[indices]
            task_y_train = y_train[indices]

            current = sklearn.ensemble.RandomForestRegressor(n_estimators=64)
            current.fit(task_X_train, task_y_train.flatten())
            self.models[task_idx] = current

    def predict(self, X_test):
        all_tasks = np.unique(X_test[:, -1])
        assert len(all_tasks) == 1, 'Can only predict for one task'

        return self.models[all_tasks[0]].predict(X_test)
