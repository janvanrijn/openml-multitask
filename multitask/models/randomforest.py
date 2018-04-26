import GPy
import numpy as np

class MetaRandomForestRegressor(object):

    def __init__(self):
        self.m = None
        self.name = 'RandomForest'

    def fit(self, task_X_train, task_y_train):
        raise NotImplementedError()

    def predict(self, task_X_test, idx):
        raise NotImplementedError()
