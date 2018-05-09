import george
import multitask.utils
import numpy as np

import robo.models.mtbo_gp
import robo.priors.env_priors


class MetaMultitaskGPGeorgeOffgrid(object):

    def __init__(self, lower_bounds, upper_bounds):
        self.name = 'MultitaskGPGeorge'
        self.model = None
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

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
        rng = np.random.RandomState(42)

        num_tasks = len(np.unique(X_train[:, -1]))
        num_obs, num_feats = X_train.shape
        num_feats -= 1
        assert len(self.lower_bounds) == len(self.upper_bounds) == num_feats

        n_hypers = 20
        chain_length = 200
        burnin = 100

        # Define model for the objective function
        cov_amp = 1  # Covariance amplitude
        kernel = cov_amp

        # ARD Kernel for the configuration space
        for d in range(num_feats):
            kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                    ndim=num_feats + 1, dim=d)

        task_kernel = george.kernels.TaskKernel(num_feats + 1, num_feats, num_tasks)
        kernel *= task_kernel

        # Take 3 times more samples than we have hyperparameters
        if n_hypers < 2 * len(kernel):
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

        prior = robo.priors.env_priors.MTBOPrior(len(kernel) + 1,
                                                 n_ls=num_feats,
                                                 n_kt=len(task_kernel),
                                                 rng=rng)

        self.model = robo.models.mtbo_gp.MTBOGPMCMC(kernel,
                                                    prior=prior,
                                                    burnin_steps=burnin,
                                                    chain_length=chain_length,
                                                    n_hypers=n_hypers,
                                                    lower=self.lower_bounds,
                                                    upper=self.upper_bounds,
                                                    rng=rng)
        self.model.train(X_train, y_train.reshape((len(y_train),)), do_optimize=True)

    def predict(self, X_test):
        """
        Predicts for new tasks

        :param task_X_test: a nd array with shape (n_obs, n_feats + 1),
        where the last feature column indicates the task
        :return:
        """
        num_obs, _ = X_test.shape
        output_index = X_test[:, -1]

        mean, variance = self.model.predict(X_test)
        return mean.flatten()
