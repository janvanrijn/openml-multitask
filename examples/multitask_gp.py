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
    parser.add_argument('--cache_directory', type=str, default='C:/experiments/multitask/cache/')
    parser.add_argument('--plot_directory', type=str, default='C:/experiments/multitask/multi/')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--optimization_method', type=str, default='Nelder-Mead')
    parser.add_argument('--maxiter', type=int, default=None)

    return parser.parse_args()


def compute_Sigma(x, M, K_f, sigma_l_2, Theta_x):
    N = len(x) # num data points
    if len(Theta_x) != 2:
        raise ValueError()
    if sigma_l_2.shape != (M, ):
        raise ValueError()

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


def do_inference(k_l_f, Sigma_inv, x, bold_y, x_star):
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
    variance = multitask.utils.rbf_kernel1D(x_star, x_star) - k_combined.T.dot(Sigma_inv).dot(k_combined)
    if not isinstance(f_l_bar, float):
        raise ValueError('predictive mean should be a scalar')
    if not isinstance(variance, float):
        raise ValueError('predictive variance should be a scalar')
    return f_l_bar, variance


def neg_log_likelihood(parameters, x, Y):
    # this function is used by scipy.optimize.minimize(). Therefore, the
    # parameters to be optimized are wrapped in the single argument
    # 'parameters', which is an array of floats. Contains
    # - the Cholesky decomposition of K_f (size: len(Y)^2)
    # - Theta_x (size: 2) # TODO: Assumption: we use the same theta for each task
    N = len(x)
    M = len(Y)
    L, sigma_l_2, Theta_x = multitask.utils.unpack_params(parameters, M)
    K_f = L.dot(L.T)

    bold_y = np.reshape(Y, (Y.shape[0] * Y.shape[1]))
    Sigma = compute_Sigma(x, M, K_f, sigma_l_2, Theta_x)
    expected_shape_sigma = (N * M, N * M)
    if Sigma.shape != expected_shape_sigma:
        raise ValueError('Wrong shape of Sigma. Expected %s got %s' %(Sigma.shape, expected_shape_sigma))
    Sigma_inv = np.linalg.inv(Sigma)

    sum_log_likelihood = 0.0
    for task_l in range(M):
        for x_idx in range(N):
            mu, _ = do_inference(K_f[:, task_l], Sigma_inv, x, bold_y, x[x_idx])
            # TODO: assumption: we could use sigma (co-variance matrix) here (instead of predictive variance)
            current_loglikelyhood = scipy.stats.norm(mu, Sigma[x_idx][x_idx]).logpdf(Y[task_l][x_idx])
            sum_log_likelihood += current_loglikelyhood
    return -1 * sum_log_likelihood


def plot_model(x_train, Y_train, task_l, K_f, sigma_l_2, Theta_x, plot_offset, target_name):
    # TODO: code duplication!
    M = len(Y_train)
    bold_y = np.reshape(Y_train, (Y_train.shape[0] * Y_train.shape[1]))
    Sigma = compute_Sigma(x_train, M, K_f, sigma_l_2, Theta_x)
    Sigma_inv = np.linalg.inv(Sigma)

    x_vals = np.linspace(min(x_train) - plot_offset, max(x_train) + plot_offset, 500)
    predictions = []
    errorbar_low = []
    errorbar_up = []
    for x_val in range(len(x_vals)):
        mu, variance = do_inference(K_f[:, task_l], Sigma_inv, x_train, bold_y, x_val * 1.0)
        stdev = variance ** 0.5

        predictions.append(mu)
        errorbar_low.append(mu - stdev)
        errorbar_up.append(mu + stdev)

    fig, ax = plt.subplots()
    ax.fill_between(x_vals, errorbar_low, errorbar_up, color="#dddddd")
    ax.plot(x_vals, predictions, 'r--', lw=2)
    ax.plot(x_train, Y_train[task_l], 'bs', ms=4)

    ax.set_ylim([0., 1.])
    fig.savefig(fname=target_name)


def optimize(x_train, Y_train, optimization_method, maxiter):
    def log_iteration(current_params):
        global optimization_steps
        optimization_steps += 1
        print('Evaluated:', optimization_steps, current_params)

    M = len(Y_train)
    optimizee = functools.partial(neg_log_likelihood, x=x_train, Y=Y_train)
    # assumptions: we have to learn the variance of the tasks (sigma_l_2)
    params0 = multitask.utils.pack_params(np.tril(np.random.rand(M, M)), np.random.rand(M), np.random.rand(2))

    options = dict()
    if maxiter:
        options['maxiter'] = maxiter

    result = scipy.optimize.minimize(optimizee, params0,
                                     method=optimization_method,
                                     callback=log_iteration,
                                     options=options)
    return result


def optimize_decorator(x_train, Y_train, optimization_method, maxiter, cahce_directory):
    if cahce_directory[-1] != '/':
        raise ValueError('Cache directory should have tailing slash')

    fn_hash = str(hash(tuple(x_train))) + '__' + \
              str(hash(tuple(map(tuple, Y_train)))) + '__' + \
              optimization_method + '__' + \
              str(maxiter)
    filepath = cahce_directory + fn_hash + '.pkl'

    if os.path.isfile(filepath):
        print('Optimization result obtained from cache..')
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)

    try:
        os.mkdir(cahce_directory)
    except FileExistsError:
        pass

    result = optimize(x_train, Y_train, optimization_method, maxiter)
    with open(filepath, 'wb') as fp:
        pickle.dump(result, fp)
    return result


def run(data_filepath, x_column, optimization_method, maxiter, cache_directory, plot_directory):
    with open(data_filepath, 'r') as fp:
        dataset = arff.load(fp)
    x_idx = None
    y_indices = []
    for idx, (column, type) in enumerate(dataset['attributes']):
        if column == x_column:
            x_idx = idx
        else:
            y_indices.append(idx)

    if x_idx is None:
        raise ValueError('Couldn\'t find x column: %s' %x_column)

    data = np.array(dataset['data'])
    x_train = data[:,x_idx]
    Y_train = np.array([data[:,y_idx] for y_idx in y_indices])

    # convert ndarrays to tuples for lru_cache
    result = optimize_decorator(x_train, Y_train, optimization_method, maxiter, cache_directory)
    parameters = result.x
    L, sigma_l_2, Theta_x = multitask.utils.unpack_params(parameters, len(Y_train))
    K_f = L.dot(L.T)

    for i in range(len(Y_train)):
        plot_model(x_train, Y_train, i, K_f, sigma_l_2, Theta_x, plot_offset=3, target_name=plot_directory + 'multitask-%d.png' %i)


if __name__ == '__main__':
    args = parse_args()
    optimization_steps = 0
    run(args.data_file, args.x_column, args.optimization_method, args.maxiter, args.cache_directory, args.plot_directory)
