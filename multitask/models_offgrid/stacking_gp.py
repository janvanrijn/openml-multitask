import GPy

import multitask.utils
import numpy as np


class MetaStackingGPOffgrid(object):

    def __init__(self):
        self.models = dict()
        self.name = 'StackingGP'
        self.X_train = None
        self.y_train = None

    def get_name(self, num_tasks, num_obs):
        return '%s.%d.%d' % (self.name, num_tasks, num_obs)

    def _add_individual_predictions(self, task_X_train, test_task_id):
        individual_predictions = None
        for idx, model in enumerate(self.models):
            if idx == test_task_id:
                continue
            mean, _ = self.models[idx].predict(task_X_train)

            if individual_predictions is None:
                individual_predictions = mean
            else:
                individual_predictions = np.concatenate((individual_predictions, mean), axis=1)
        assert individual_predictions.shape == (len(task_X_train), len(self.models) - 1)
        task_X_train = np.concatenate((task_X_train, individual_predictions), axis=1)
        return task_X_train

    @multitask.utils.fit_and_measure_time
    def fit(self, X_train, y_train):
        num_obs, num_feats = X_train.shape
        assert y_train.shape == (num_obs, 1)

        all_tasks = np.unique(X_train[:, -1])
        for task_idx in list(all_tasks):
            indices = X_train[:, -1] == task_idx
            task_X_train = X_train[indices]
            task_y_train = y_train[indices]

            kernel = GPy.kern.RBF(input_dim=num_feats, ARD=True)
            current = GPy.models.GPRegression(task_X_train, task_y_train, kernel)
            current.optimize()
            self.models[task_idx] = current
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        all_tasks = np.unique(X_test[:, -1])
        assert len(all_tasks) == 1, 'Can only predict for one task'
        test_task_id = all_tasks[0]

        indices = self.X_train[:, -1] == test_task_id
        task_X_train = self.X_train[indices]
        task_y_train = self.y_train[indices]
        task_X_train = self._add_individual_predictions(task_X_train, test_task_id)

        kernel = GPy.kern.RBF(input_dim=task_X_train.shape[1], ARD=True)
        stacking_model = GPy.models.GPRegression(task_X_train, task_y_train, kernel)
        stacking_model.optimize()

        mean, variance = stacking_model.predict(self._add_individual_predictions(X_test, test_task_id))
        return mean.flatten()

    def plot(self, idx, axes):
        self.models[idx].plot(ax=axes, fixed_inputs=[(1, idx)], plot_data=False)
