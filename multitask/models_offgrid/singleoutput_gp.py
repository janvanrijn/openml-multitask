import GPy
import numpy as np


class MetaSingleOutputGPOffgrid(object):

    def __init__(self):
        self.models = dict()
        self.name = 'SingleOutputGP'

    def get_name(self, num_tasks, num_obs):
        return self.name

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

    def predict(self, X_test):
        all_tasks = np.unique(X_test[:, -1])
        assert len(all_tasks) == 1, 'Can only predict for one task'

        mean, variance = self.models[all_tasks[0]].predict(X_test)
        return mean.flatten()

    def plot(self, idx, axes):
        self.models[idx].plot(ax=axes, fixed_inputs=[(1, idx)], plot_data=False)
