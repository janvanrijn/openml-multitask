import GPy


class MetaSingleOutputGPOngrid(object):

    def __init__(self):
        self.m = list()
        self.name = 'SingleOutputGP'

    def fit(self, X_train, Y_train):
        num_obs, num_feats = X_train.shape
        _, num_tasks = Y_train.shape

        for idx in range(num_tasks):
            kernel = GPy.kern.RBF(input_dim=num_feats, variance=1., lengthscale=1.)
            current_Y = Y_train[:, idx].reshape((num_obs, 1))
            current = GPy.models.GPRegression(X_train, current_Y, kernel)
            current.optimize()
            self.m.append(current)

    def predict(self, X_test, idx):
        mean, variance = self.m[idx].predict(X_test)
        return mean.flatten()
