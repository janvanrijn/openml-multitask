import sklearn.ensemble


class MetaRandomForestRegressor(object):

    def __init__(self):
        self.m = list()
        self.name = 'RandomForest'

    def fit(self, task_X_train, task_y_train):
        num_tasks, _, _ = task_X_train.shape
        for idx in range(num_tasks):
            current = sklearn.ensemble.RandomForestRegressor(n_estimators=64)
            current.fit(task_X_train[idx], task_y_train[idx].flatten())
            self.m.append(current)

    def predict(self, task_X_test, idx):
        return self.m[idx].predict(task_X_test[idx])
