import argparse
import arff
import functools
import matplotlib.pyplot as plt
import multitask
import numpy as np
import os
import pickle
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--use_cache', action='store_true', default=False)
    parser.add_argument('--cache_directory', type=str, default='C:/experiments/multitask/cache/')
    parser.add_argument('--plot_directory', type=str, default='C:/experiments/multitask/multi/')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--optimization_method', type=str, default='Nelder-Mead')
    parser.add_argument('--maxiter', type=int, default=None)

    return parser.parse_args()


def compute_Sigma(x, M, K_f, sigma_l_2, Theta_x):
    N = len(x) # num data points
    if len(Theta_x) != 2:
        raise ValueError()
    if sigma_l_2.shape != (M, ):
        raise ValueError('Incorrect shape of sigma_l_2. Expected (%d, ) got: %s' %(M, sigma_l_2.shape))

    I = np.eye(N)
    D = np.zeros((M, M))
    D[np.diag_indices(M)] = sigma_l_2

    K_x = multitask.utils.rbf_kernel1D(x, x, Theta_x[0], Theta_x[1])
    if K_x.shape != (N, N):
        raise ValueError()
    Sigma = np.kron(K_f, K_x) + np.kron(D, I)
    if Sigma.shape != (N*M, N*M):
        raise ValueError()
    return Sigma


def do_inference(k_l_f, task_l, Sigma_inv, x, bold_y, x_star):
    if not isinstance(x_star, float):
        raise ValueError('x_star should be scalar, got: %s' %x_star)
    if not Sigma_inv.shape[0] == Sigma_inv.shape[1]:
        raise ValueError('Sigma_inv should be squared')
    if not (Sigma_inv.shape[1],) == bold_y.shape:
        raise ValueError('Sigma_inv and bold_y dimensions should match')

    k_s_x = multitask.utils.rbf_kernel1D(x_star, x)
    if k_s_x.shape != (len(x), ):
        raise ValueError('Wrong dimensionality of k_s_x (check the kernel)')

    k_combined = np.kron(k_l_f, k_s_x)
    f_l_bar = k_combined.T.dot(Sigma_inv).dot(bold_y)
    # TODO: calculating variance based on Rasmussen Eq. 2.26.
    # TODO: question: How to incorporate data kernel in first part of equation (part before minus operator)?
    variance = np.kron(k_l_f[task_l], multitask.utils.rbf_kernel1D(x_star, x_star)) - k_combined.T.dot(Sigma_inv).dot(k_combined)

    if not isinstance(f_l_bar, float):
        raise ValueError('predictive mean should be a scalar')
    if not isinstance(variance, float):
        raise ValueError('predictive variance should be a scalar')
    return f_l_bar, variance


def neg_log_likelihood(parameters, x, Y_train):
    # this function is used by scipy.optimize.minimize(). Therefore, the
    # parameters to be optimized are wrapped in the single argument
    # 'parameters', which is an array of floats. Contains
    # - the Cholesky decomposition of K_f (size: len(Y_train)^2)
    # - Theta_x (size: 2) # TODO: Assumption: we use the same theta for each task
    N, M = Y_train.shape
    L, Theta_x, sigma_l_2 = multitask.utils.unpack_params(parameters, M, include_sigma=True, include_theta=True)
    K_f = L.dot(L.T)

    bold_y = np.reshape(Y_train.T, (N * M)).T
    Sigma = compute_Sigma(x, M, K_f, sigma_l_2, Theta_x)
    expected_shape_sigma = (N * M, N * M)
    if Sigma.shape != expected_shape_sigma:
        raise ValueError('Wrong shape of Sigma. Expected %s got %s' %(Sigma.shape, expected_shape_sigma))
    # if np.linalg.det(Sigma) == 0:
    #     raise ValueError('Sigma has determinant of zero. ')
    Sigma_inv = np.linalg.inv(Sigma)

    lml = multitask.utils.inference._marginal_likelihood(Sigma, Sigma_inv, bold_y)
    return -1 * lml


def plot_model(x_train, Y_train, task_l, K_f, sigma_l_2, Theta_x, plot_offset, param_name, target_name):
    N, M = Y_train.shape
    bold_y = np.reshape(Y_train.T, (N * M)).T
    Sigma = compute_Sigma(x_train, M, K_f, sigma_l_2, Theta_x)
    Sigma_inv = np.linalg.inv(Sigma)

    x_vals = np.linspace(min(x_train) - plot_offset, max(x_train) + plot_offset, 500)
    predictions = []
    errorbar_low = []
    errorbar_up = []
    for x_val in x_vals:
        mu, variance = do_inference(K_f[:, task_l], task_l, Sigma_inv, x_train, bold_y, x_val * 1.0)
        stdev = variance ** 0.5

        predictions.append(mu)
        errorbar_low.append(mu - stdev)
        errorbar_up.append(mu + stdev)

    fig, ax = plt.subplots()
    ax.fill_between(x_vals, errorbar_low, errorbar_up, color="#dddddd")
    ax.plot(x_vals, predictions, 'r--', lw=2)
    ax.plot(x_train, Y_train[:, task_l], 'bs', ms=4)
    ax.set(xlabel=param_name, ylabel='predictive_accuracy',
           title='Multi Task GP on ' + target_name)

    ax.set_ylim([0., 1.])
    fig.savefig(fname=target_name)


def optimize(x_train, Y_train, optimization_method, maxiter):
    def log_iteration(current_params):
        global optimization_steps
        optimization_steps += 1
        print('Evaluated:', optimization_steps, current_params)

    optimizee = functools.partial(neg_log_likelihood, x=x_train, Y_train=Y_train)
    # assumptions: we have to learn the variance of the tasks (sigma_l_2)
    N, M = Y_train.shape
    K_x = multitask.utils.rbf_kernel1D(x_train, x_train)
    K_x_inv = np.linalg.inv(K_x)
    K_f_init = 1 / N * Y_train.T.dot(K_x_inv).dot(Y_train)
    K_f_init_inv = np.linalg.cholesky(K_f_init)
    params0 = multitask.utils.pack_params(np.tril(K_f_init_inv), [1.0, 1.0], np.zeros(M))

    options = dict()
    if maxiter:
        options['maxiter'] = maxiter

    result = scipy.optimize.minimize(optimizee, params0,
                                     method=optimization_method,
                                     callback=log_iteration,
                                     options=options)

    return result


def optimize_decorator(x_train, Y_train, optimization_method, maxiter, use_cache, cahce_directory):
    if cahce_directory[-1] != '/':
        raise ValueError('Cache directory should have tailing slash')

    fn_hash = str(hash(tuple(x_train))) + '__' + \
              str(hash(tuple(map(tuple, Y_train)))) + '__' + \
              optimization_method + '__' + \
              str(maxiter)
    filepath = cahce_directory + fn_hash + '.pkl'

    if os.path.isfile(filepath) and use_cache:
        print('Optimization result obtained from cache..')
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)

    try:
        os.mkdir(cahce_directory)
    except FileExistsError:
        pass

    result = optimize(x_train, Y_train, optimization_method, maxiter)
    if use_cache:
        with open(filepath, 'wb') as fp:
            pickle.dump(result, fp)
    return result


def run(args):
    with open(args.data_file, 'r') as fp:
        dataset = arff.load(fp)
    x_idx = None
    y_indices = []
    for idx, (column, type) in enumerate(dataset['attributes']):
        if column == args.x_column:
            x_idx = idx
        elif args.max_tasks is None or len(y_indices) < args.max_tasks:
            y_indices.append(idx)

    if x_idx is None:
        raise ValueError('Couldn\'t find x column: %s' %args.x_column)

    data = np.array(dataset['data'])
    x_train = data[:,x_idx]
    Y_train = np.array([data[:, y_idx] for y_idx in y_indices]).T
    N, M = Y_train.shape

    result = optimize_decorator(x_train, Y_train, args.optimization_method, args.maxiter, args.use_cache, args.cache_directory)
    incumbent = result.x
    L, Theta_x, sigma_l_2 = multitask.utils.unpack_params(incumbent, M, include_sigma=True, include_theta=True)
    K_f = L.dot(L.T)
    # TODO: why is K_f not equal to [[1]] when we use just one task (i.e., M=1, see email to Andreas on March 29)

    for idx, y_column in enumerate(y_indices):
        current_target = dataset['attributes'][y_column][0]
        print(current_target)
        plot_model(x_train, Y_train, idx, K_f, sigma_l_2, Theta_x,
                   plot_offset=3, param_name=dataset['attributes'][x_idx][0],
                   target_name=args.plot_directory + current_target + '.png')


if __name__ == '__main__':
    # used by optimizer to keep a count
    optimization_steps = 0
    run(parse_args())
