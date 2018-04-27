import GPy


class MetaMultiOutputGPOngrid(object):

    def __init__(self):
        self.m = None
        self.name = 'MultiOutputGP'

    def fit(self, X_train, Y_train):

        self.m = GPy.models.gp_multiout_regression.GPMultioutRegression(X_train, Y_train, Xr_dim=Y_train.shape[1])
        self.m.optimize()

    def predict(self, X_test, idx):
        mean, variance = self.m.predict(X_test)
        return mean[:, idx]
