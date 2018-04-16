import argparse
import arff
import functools
import matplotlib.pyplot as plt
import multitask
import numpy as np
import os
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--plot_directory', type=str, default='C:/experiments/multitask/single/')
    parser.add_argument('--default_value_theta0', type=float, default=1.0)
    parser.add_argument('--default_value_theta1', type=float, default=1.0)
    parser.add_argument('--optimization_method', type=str, default='Nelder-Mead')
    parser.add_argument('--maxiter', type=int, default=None)

    return parser.parse_args()


def plot(x_train, y_train, x, Theta_x, plot_directory, target_name, param_name, num_samples=3):
    fig, ax = plt.subplots()

    mu, sigma, _ = multitask.utils.get_posterior(x, x_train, y_train, Theta_x[0], Theta_x[1])

    variance = sigma.diagonal()
    ax.fill_between(x, mu - variance ** 0.5, mu + variance ** 0.5, color="#dddddd")
    ax.plot(x, mu, 'r--', lw=2)

    if num_samples:
        for i in range(num_samples):
            ax.plot(x, np.random.multivariate_normal(mu, sigma))

    ax.plot(x_train, y_train, 'bs', ms=4)
    ax.set(xlabel=param_name, ylabel='predictive_accuracy',
           title='Single Task GP on %s Theta = %s' %(target_name, Theta_x) )

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0., 1.])
    output_file = os.path.join(plot_directory, target_name + '.png')
    fig.savefig(fname=output_file)


def neg_log_likelihood(Theta_x, x_train, y_train):
    _, _, lml = multitask.utils.get_posterior(x_train, x_train, y_train, Theta_x[0], Theta_x[1])
    return -1 * lml


def optimize_theta(x, y, default_value_theta0, default_value_theta1, optimization_method, maxiter):
    optimizee = functools.partial(neg_log_likelihood, x_train=x, y_train=y)
    options = dict()
    if maxiter:
        options['maxiter'] = maxiter
    result = scipy.optimize.minimize(optimizee,
                                     [default_value_theta0, default_value_theta1],
                                     method=optimization_method,
                                     options=options)
    return result.x


def run(args):
    plot_offset = 3
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
    x = data[:,x_idx]
    x_star = np.linspace(min(x) - plot_offset, max(x) + plot_offset, 500)

    for y_idx in y_indices:
        current_target = dataset['attributes'][y_idx][0]
        y = data[:, y_idx]

        Theta_x = optimize_theta(x, y, args.default_value_theta0, args.default_value_theta1, args.optimization_method, args.maxiter)
        print(current_target, Theta_x)

        plot(x, y, x_star, Theta_x, plot_directory=args.plot_directory, target_name=current_target, param_name=args.x_column)


if __name__ == '__main__':
    run(parse_args())
