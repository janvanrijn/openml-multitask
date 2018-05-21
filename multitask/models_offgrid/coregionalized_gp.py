import GPy
import multitask.utils
import numpy as np
import os
import pickle


class MetaCoregionalizedGPOffgrid(object):

    def __init__(self):
        self.name = 'CoregionalizedGP'
        self.model = None
        self.train_time = None

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
        num_tasks = len(np.unique(X_train[:, -1]))
        num_obs, num_feats = X_train.shape
        assert y_train.shape == (num_obs, 1)

        kern = GPy.kern.RBF(input_dim=num_feats-1, ARD=True) ** \
               GPy.kern.Coregionalize(input_dim=1, output_dim=num_tasks, rank=1)
        self.model = GPy.models.GPRegression(X_train, y_train, kern)
        self.model.optimize()

    def predict(self, X_test):
        """
        Predicts for new tasks

        :param task_X_test: a nd array with shape (n_obs, n_feats + 1),
        where the last feature column indicates the task
        :return:
        """
        num_obs, _ = X_test.shape
        output_index = X_test[:, -1]

        mean, variance = self.model.predict(X_test, Y_metadata=output_index)
        return mean.flatten()

    def plot(self, idx, axes):
        self.model.plot(ax=axes, fixed_inputs=[(1, idx)], plot_data=False)

    # def serialize(self, folder):
    #     os.makedirs(folder, exist_ok=True)
    #     if self.model is None:
    #         raise ValueError('Model not fitted yet.')
    #     self.model.save_model(os.path.join(folder, 'model.gpy'), compress=True, save_data=False)
    #     with open(os.path.join(folder, 'meta.pkl'), 'wb') as fp:
    #         pickle.dump({'train_time': self.train_time}, fp)
    #
    # @classmethod
    # def deserialize(cls, folder):
    #     with open(os.path.join(folder, 'meta.pkl'), 'rb') as fp:
    #         meta = pickle.load(fp)
    #
    #     inst = cls()
    #     inst.train_time = meta.train_time
    #     inst.model = GPy.models.GPRegression.load_model(os.path.join(folder, 'model.gpy'))
