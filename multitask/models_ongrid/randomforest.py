import sklearn.ensemble


class MetaRandomForestOngrid(object):

    def __init__(self):
        self.m = list()
        self.name = 'RandomForest'

    def fit(self, X_train, Y_train):
        _, num_tasks = Y_train.shape
        for idx in range(num_tasks):
            current = sklearn.ensemble.RandomForestRegressor(n_estimators=64)
            current.fit(X_train, Y_train[:, idx])
            self.m.append(current)

    def predict(self, X_test, idx):
        return self.m[idx].predict(X_test)
