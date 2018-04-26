import GPy


class MetaGaussianProcessOffgrid(object):

    def __init__(self):
        self.m = list()
        self.name = 'Singletask GP'

    def fit(self, task_X_train, task_y_train):
        num_tasks, _, num_feats = task_X_train.shape
        for idx in range(num_tasks):
            kernel = GPy.kern.RBF(input_dim=num_feats, variance=1., lengthscale=1.)
            current = GPy.models.GPRegression(task_X_train[idx], task_y_train[idx], kernel)
            current.optimize()
            self.m.append(current)

    def predict(self, task_X_test, idx):
        mean, variance = self.m[idx].predict(task_X_test[idx])
        return mean.flatten()
