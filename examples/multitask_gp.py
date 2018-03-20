import argparse
import arff
import functools
import multitask
import numpy as np
import os
import pickle
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--cache_directory', type=str, default='C:/experiments/multitask/cache/')
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

    f_l_bar = np.kron(k_l_f, k_s_x).T.dot(Sigma_inv).dot(bold_y)
    # TODO: assumption by jvR this returns the predictive mean, i.e., mu
    # TODO: question: How to calculate stdev sigma?
    # TODO: assumption: Or can I just plug covariance matrix Sigma in normal dist obj?
    # -- this is weird because Sigma does not depend in any way on the x_star and no guarantees the size makes sense
    return f_l_bar


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
            mu = do_inference(K_f[:, task_l], Sigma_inv, x, bold_y, x[x_idx])
            # TODO: assumption: we could use sigma (co-variance matrix) here
            current_loglikelyhood = scipy.stats.norm(mu, Sigma[x_idx][x_idx]).logpdf(Y[task_l][x_idx])
            # TODO: check scalar
            sum_log_likelihood += current_loglikelyhood
    return -1 * sum_log_likelihood


def apply_model(x_train, Y_train, x_test, task_l, K_f, sigma_l_2, Theta_x):
    # TODO: code duplication!
    M = len(Y_train)
    bold_y = np.reshape(Y_train, (Y_train.shape[0] * Y_train.shape[1]))
    Sigma = compute_Sigma(x_train, M, K_f, sigma_l_2, Theta_x)
    Sigma_inv = np.linalg.inv(Sigma)

    predictions = []
    for x_idx in range(len(x_test)):
        mu = do_inference(K_f[:, task_l], Sigma_inv, x_train, bold_y, x_test[x_idx])
        predictions.append(mu)
    return predictions


def optimize(x_train, Y_train, optimization_method, maxiter):
    def log_iteration(current_params):
        global optimization_steps
        optimization_steps += 1
        print('Evaluated:', optimization_steps, current_params)

    M = len(Y_train)
    optimizee = functools.partial(neg_log_likelihood, x=x_train, Y=Y_train)
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


def run(data_filepath, x_column, optimization_method, maxiter, cache_directory):
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
    x = data[:,x_idx]
    Y = np.array([data[:,y_idx] for y_idx in y_indices])

    x_train = x[range(0, len(x), 2)]
    Y_train = Y[:, range(0, len(x), 2)]

    x_test = x[range(1, len(x), 2)]
    Y_test = Y[:, range(1, len(x), 2)]

    # convert ndarrays to tuples for lru_cache
    result = optimize_decorator(x_train, Y_train, optimization_method, maxiter, cache_directory)
    print(result)
    parameters = result.x
    L, sigma_l_2, Theta_x = multitask.utils.unpack_params(parameters, len(Y))
    K_f = L.dot(L.T)
    for i in range(len(Y)):
        predictions = apply_model(x_train, Y_train, x_test, i, K_f, sigma_l_2, Theta_x)
        print(predictions)


if __name__ == '__main__':
    args = parse_args()
    optimization_steps = 0
    run(args.data_file, args.x_column, args.optimization_method, args.maxiter, args.cache_directory)
