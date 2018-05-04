import GPy
import numpy as np


class MetaCoregionalizedGPOffgrid(object):

    def __init__(self):
        self.name = 'CoregionalizedGP'
        self.model = None

    def fit(self, task_X_train, task_y_train):
        num_tasks, num_obs, num_feats = task_X_train.shape

        X_train = np.reshape(task_X_train, (num_tasks * num_obs, num_feats))
        Y_train = np.reshape(task_y_train, (num_tasks * num_obs, 1))

        kern = GPy.kern.RBF(input_dim=num_feats-1, ARD=True) ** \
               GPy.kern.Coregionalize(input_dim=1, output_dim=num_tasks, rank=1)
        self.model = GPy.models.GPRegression(X_train, Y_train, kern)
        self.model.optimize()

    def predict(self, task_X_test, idx):
        _, num_observations, _ = task_X_test.shape
        output_index = np.full((num_observations, 1), idx)

        mean, variance = self.model.predict(task_X_test[idx], Y_metadata=output_index)
        return mean.flatten()

    def plot(self, idx, axes):
        self.models.plot(ax=axes, fixed_inputs=[(1, idx)], plot_data=False)
