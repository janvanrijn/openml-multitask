import multitask.utils
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


class MetaCoregionalizedRFOffgrid(object):

    def __init__(self):
        self.name = 'CoregionalizedRF'
        self.model = None

    def get_name(self, num_tasks, num_obs):
        return '%s.%d.%d' % (self.name, num_tasks, num_obs)

    @multitask.utils.fit_and_measure_time
    def fit(self, X_train, y_train):
        """
        Trains the model

        :param task_X_train: a nd array with shape (n_obs, n_feats + 1),
        where the last feature column indicates the task
        :param task_y_train: a nd array of the shape (n_obs, 1)
        """
        num_obs, num_feats = X_train.shape
        assert y_train.shape == (num_obs, 1)

        self.model = sklearn.pipeline.Pipeline(steps=[
            ('hotencoding', sklearn.preprocessing.OneHotEncoder(categorical_features=[num_feats-1])),  # last feature indicates task
            ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=64))
        ])
        self.model.fit(X_train, y_train.flatten())

    def predict(self, task_X_test):
        return self.model.predict(task_X_test)
