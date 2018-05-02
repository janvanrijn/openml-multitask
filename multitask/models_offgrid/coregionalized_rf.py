import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


class MetaCoregionalizedRFOffgrid(object):

    def __init__(self):
        self.name = 'CoregionalizedRF'
        self.model = None

    def fit(self, task_X_train, task_y_train):
        num_tasks, num_obs, num_feats = task_X_train.shape

        X_train = np.reshape(task_X_train, (num_tasks * num_obs, num_feats))
        y_train = np.reshape(task_y_train, (num_tasks * num_obs))

        self.model = sklearn.pipeline.Pipeline(steps=[
            ('hotencoding', sklearn.preprocessing.OneHotEncoder(categorical_features=[num_feats-1])),  # last feature indicates task
            ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=64))
        ])
        self.model.fit(X_train, y_train)

    def predict(self, task_X_test, idx):
        return self.model.predict(task_X_test[idx])
