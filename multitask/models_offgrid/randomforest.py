import sklearn.ensemble


class MetaRandomForestOffgrid(object):

    def __init__(self):
        self.models = list()
        self.name = 'RandomForest'

    def fit(self, task_X_train, task_y_train):
        num_tasks, _, _ = task_X_train.shape
        for idx in range(num_tasks):
            current = sklearn.ensemble.RandomForestRegressor(n_estimators=64)
            current.fit(task_X_train[idx], task_y_train[idx].flatten())
            self.models.append(current)

    def predict(self, task_X_test, idx):
        return self.models[idx].predict(task_X_test[idx])
