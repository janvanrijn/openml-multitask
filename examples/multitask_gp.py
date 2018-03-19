import argparse
import arff
import functools
import multitask
import numpy as np
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--maxiter', type=int, default=100)

    return parser.parse_args()


def compute_Sigma(x, M, K_f, Theta_x):
    if len(Theta_x) != 2:
        raise ValueError()

    N = len(x) # num data points
    I = np.eye(N)
    D = np.zeros((M, M))
    # TODO: the diagonal of D should be sigma_l^2
    # p2: "sigma_l^2 is the noice variance for the l^th task"
    # (7/7/18) assumption by AM: sigma_l^2 is learned
    # TODO 2: assumption, D can be all zeros, for "noiseless observations"
    # TODO 3: question what is the range of sigma_L^2

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
    K_f, Theta_x = multitask.utils.unpack_params(parameters, M)

    bold_y = np.reshape(Y, (Y.shape[0] * Y.shape[1]))
    Sigma = compute_Sigma(x, M, K_f, Theta_x)
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


def run(data_filepath, x_column, maxiter):
    def log_iteration(current_params):
        global optimization_steps
        optimization_steps += 1
        print('Evaluated:', optimization_steps, current_params)

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
    optimizee = functools.partial(neg_log_likelihood, x=x, Y=Y)
    params0 = multitask.utils.pack_params(np.eye(len(Y)), [0.1, 0.1])

    optimizee(params0)

    options = dict()
    if maxiter:
        options['maxiter'] = maxiter

    # result = scipy.optimize.minimize(optimizee, params0,
    #                                  method='BFGS',
    #                                  callback=log_iteration,
    #                                  options=options)
    # print(result)


if __name__ == '__main__':
    args = parse_args()
    optimization_steps = 0
    run(args.data_file, args.x_column, args.maxiter)
