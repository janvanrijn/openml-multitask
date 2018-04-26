import GPy
import numpy as np


class MetaMultiOutputGPOngrid(object):

    def __init__(self):
        self.m = None
        self.name = 'GridRegression'

    def fit(self, X_train, Y_train):
        print(X_train.shape)
        print(Y_train.shape)
        kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1, ARD=True)
        self.m = GPy.models.gp_grid_regression.GPRegressionGrid(X_train, Y_train, kernel=kernel)
        self.m.optimize()

    def predict(self, X_test, idx):
        num_obs, _ = X_test.shape
        output_index = np.full((num_obs, 1), idx)

        return self.predict(X_test, Y_metadata={'output_index': output_index})
