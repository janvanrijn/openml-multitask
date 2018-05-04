import GPy
import numpy as np

class MetaSingleOutputGPOffgrid(object):

    def __init__(self):
        self.models = list()
        self.name = 'SingleOutputGP'

    def fit(self, task_X_train, task_y_train):
        num_tasks, num_obs, num_feats = task_X_train.shape
        for idx in range(num_tasks):

            y_train = np.reshape(task_y_train[idx], (num_obs, 1))
            kernel = GPy.kern.RBF(input_dim=num_feats, ARD=True)
            current = GPy.models.GPRegression(task_X_train[idx], y_train, kernel)
            current.optimize()
            self.models.append(current)

    def predict(self, task_X_test, idx):
        mean, variance = self.models[idx].predict(task_X_test[idx])
        return mean.flatten()

    def plot(self, idx, axes):
        self.models[idx].plot(ax=axes, fixed_inputs=[(1, idx)], plot_data=False)
