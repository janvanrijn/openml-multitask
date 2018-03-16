import argparse
import arff
import multitask
import numpy as np
import scipy


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')

    return parser.parse_args()


def compute_Sigma(x, Y, K_f, Theta_x):
    if len(Theta_x) != 2:
        raise ValueError()

    N = Y.shape[0] # num tasks
    M = len(x) # num data points
    I = np.eye(N)
    D = np.zeros((M, M))
    # TODO: the diagonal should be sigma_l^2
    # p2: "sigma_l^2 is the noice variance for the l^th task"
    # (7/7/18) assumption by AM: sigma_l^2 is learned
    # TODO 2: assumption, D can be all zeros, for "noiseless observations"

    K_x = multitask.utils.rbf_kernel(x, x, Theta_x[0], Theta_x[1])
    Sigma = np.kron(K_f, K_x) + np.kron(D, I)

    if Sigma.shape != (N*M, N*M):
        raise ValueError()
    return Sigma


def do_inference(k_l_f, Sigma_inv, x, y, x_star):
    # x_star is scalar
    # y is a vector here
    # TODO check inputs
    k_s_x = multitask.utils.rbf_kernel(x_star, x)
    # TODO check dimensions

    f_l_bar = np.kron(k_l_f, k_s_x).T.dot(Sigma_inv).dot(y)
    # TODO: assumption by jvR this returns the predictive mean, i.e., mu
    # TODO: question: How to calculate stdev sigma?
    # TODO: assumption: Or can I just plug covariance matrix Sigma in normal dist obj?
    # -- this is weird because Sigma does not depend in any way on the x_star and no guarantees the size makes sense
    return f_l_bar


def log_likelihood(x, Y, K_f, Theta_x):
    Sigma = compute_Sigma(x, Y, K_f, Theta_x)
    Sigma_inv = np.linalg.inv(Sigma)

    sum_log_likelihood = 0.0
    for task_l in len(Y):
        for x_idx in len(x):
            mu = do_inference(K_f[task_l], Sigma_inv, x, Y[task_l], x[x_idx])
            # TODO: should we use Sigma here?
            current_loglikelyhood = scipy.stat.norm(mu, Sigma[x_idx][x_idx]).logpdf(Y[task_l][x_idx])
            # TODO: check scalar
            sum_log_likelihood += current_loglikelyhood
    return sum_log_likelihood


def run(data_filepath, x_column):
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


if __name__ == '__main__':
    args = parse_args()
    run(args.data_file, args.x_column)