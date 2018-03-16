import argparse
import arff
import matplotlib.pyplot as plt
import multitask
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')

    return parser.parse_args()


def plot(x, mu, sigma, target_name, param_name, num_samples=25):
    fig, ax = plt.subplots()
    for i in range(num_samples):
        ax.plot(x, np.random.multivariate_normal(mu, sigma))
    ax.set(xlabel=param_name, ylabel='predictive_accuracy',
           title='GP on ' + target_name)

    ax.set_ylim([0., 1.])
    fig.savefig(fname=target_name)


def get_posterior(x_star, x, y):
    # This fn implements Eq. 2.19 from Rasmussen and Williams
    K_vv = multitask.utils.rbf_kernel(x, x)
    K_sv = multitask.utils.rbf_kernel(x_star, x)
    K_vs = multitask.utils.rbf_kernel(x, x_star)
    K_ss = multitask.utils.rbf_kernel(x_star, x_star)

    K_inv = np.linalg.inv(K_vv)

    mu = K_sv.dot(K_inv).dot(y)              # predictive means
    sigma = K_ss - K_sv.dot(K_inv).dot(K_vs) # covariance matrix (stdevs are on the diagonal)
    return mu, sigma


def run(data_filepath, x_column):
    plot_offset = 3
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
    x_star = np.linspace(min(x) - plot_offset, max(x) + plot_offset, 500)

    for y_idx in y_indices:
        current_target = dataset['attributes'][y_idx][0]
        print(current_target)
        y = data[:, y_idx]
        mu, sigma = get_posterior(x_star, x, y)
        plot(x_star, mu, sigma, target_name=current_target, param_name=x_column)


if __name__ == '__main__':
    args = parse_args()
    run(args.data_file, args.x_column)