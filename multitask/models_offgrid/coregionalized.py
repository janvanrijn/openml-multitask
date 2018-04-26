import GPy
import numpy as np


class MetaCoregionalizedGPOffgrid(object):

    def __init__(self):
        self.name = 'CoregionalizedGP'
        self.m = None

    def fit(self, task_X_train, task_y_train):
        num_tasks, _, num_feats = task_X_train.shape

        K1 = GPy.kern.Bias(num_feats)
        K2 = GPy.kern.Linear(num_feats)
        K3 = GPy.kern.Matern32(num_feats)
        lcm = GPy.util.multioutput.LCM(input_dim=num_feats, num_outputs=num_tasks, kernels_list=[K1, K2, K3])
        self.m = GPy.models.GPCoregionalizedRegression(task_X_train, task_y_train, kernel=lcm)
        self.m['.*ICM.*var'].unconstrain()
        self.m['.*ICM0.*var'].constrain_fixed(1.)
        self.m['.*ICM0.*W'].constrain_fixed(0)
        self.m['.*ICM1.*var'].constrain_fixed(1.)
        self.m['.*ICM1.*W'].constrain_fixed(0)
        self.m.optimize()

    def predict(self, task_X_test, idx):
        _, num_observations, _ = task_X_test.shape
        input_index = np.full(task_X_test[idx].shape, idx)
        output_index = np.full((num_observations, 1), idx)
        extended = np.hstack((task_X_test[idx], input_index))

        mean, variance = self.m.predict(extended, Y_metadata={'output_index': output_index})
        return mean.flatten()
