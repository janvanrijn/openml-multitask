import argparse
import arff
import functools
import matplotlib.pyplot as plt
import multitask
import numpy as np
import os
import scipy.stats
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--plot_directory', type=str, default=os.path.expanduser("~") + '/experiments/multitask/warped/')
    parser.add_argument('--default_value_theta0', type=float, default=1.0)
    parser.add_argument('--default_value_theta1', type=float, default=1.0)
    parser.add_argument('--default_value_alpha', type=float, default=5)
    parser.add_argument('--default_value_beta', type=float, default=1)
    parser.add_argument('--optimization_method', type=str, default='Nelder-Mead')
    parser.add_argument('--maxiter', type=int, default=None)
    parser.add_argument('--plot_offset', type=int, default=0)

    return parser.parse_args()


def warp_input(x, alpha, beta, range_min, range_max):
    x = scipy.stats.beta.pdf(x, a=alpha, b=beta, loc=range_min, scale=range_max-range_min)
    if range_min > min(x):
        raise ValueError('Found value=%f, with alpha=%f, beta=%f' % (min(x), alpha, beta))
    if range_max < max(x):
        raise ValueError('Found value=%f, with alpha=%f, beta=%f' % (max(x), alpha, beta))
    return x


def plot(x_train, y_train, x, params, plot_directory, target_name, param_name, range_min, range_max, num_samples=3):
    fig, ax = plt.subplots()

    x = warp_input(x, params[2], params[3], range_min, range_max)
    x_train = warp_input(x_train, params[2], params[3], range_min, range_max)

    mu, sigma, _ = multitask.utils.get_posterior(x, x_train, y_train, params[0], params[1])

    variance = sigma.diagonal()
    ax.fill_between(x, mu - variance ** 0.5, mu + variance ** 0.5, color="#dddddd")
    ax.plot(x, mu, 'r--', lw=2)

    if num_samples:
        for i in range(num_samples):
            ax.plot(x, np.random.multivariate_normal(mu, sigma))

    ax.plot(x_train, y_train, 'bs', ms=4)
    ax.set(xlabel=param_name, ylabel='predictive_accuracy',
           title='Single Task GP on %s params = %s' % (target_name, params))

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0., 1.])
    os.makedirs(plot_directory, exist_ok=True)
    output_file = os.path.join(plot_directory, target_name + '_warped.png')
    fig.savefig(fname=output_file)


def neg_log_likelihood(params, x_train, y_train, range_min, range_max):
    theta_0 = params[0]
    theta_1 = params[1]
    alpha = params[2]
    beta = params[3]
    print('=== iteration === alpha %f beta %f' % (alpha, beta))
    try:
        x_train = warp_input(x_train, alpha, beta, range_min, range_max)
        _, _, lml = multitask.utils.get_posterior(x_train, x_train, y_train, theta_0, theta_1)
    except np.linalg.LinAlgError:
        traceback.print_stack()
        return np.inf
    except ValueError:
        traceback.print_stack()
        return np.inf
    return -1 * lml


def optimize_params(x, y, default_value_theta0, default_value_theta1,
                    default_value_alpha, default_value_beta,
                    range_min, range_max,
                    optimization_method, maxiter):
    optimizee = functools.partial(neg_log_likelihood, x_train=x, y_train=y, range_min=range_min, range_max=range_max)
    options = dict()
    if maxiter:
        options['maxiter'] = maxiter
    result = scipy.optimize.minimize(optimizee,
                                     [default_value_theta0, default_value_theta1, default_value_alpha, default_value_beta],
                                     method=optimization_method,
                                     options=options)
    return result.x


def run(args):
    plot_offset = args.plot_offset
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
    x = data[:, x_idx]
    x_star = np.linspace(min(x) - plot_offset, max(x) + plot_offset, 500)

    range_min = min(x_star)
    range_max = max(x_star)

    for y_idx in y_indices:
        current_target = dataset['attributes'][y_idx][0]
        y = data[:, y_idx]

        params = optimize_params(x, y,
                                 default_value_theta0=args.default_value_theta0,
                                 default_value_theta1=args.default_value_theta1,
                                 default_value_alpha=args.default_value_alpha,
                                 default_value_beta=args.default_value_beta,
                                 range_min=range_min,
                                 range_max=range_max,
                                 optimization_method=args.optimization_method,
                                 maxiter=args.maxiter)
        print(current_target, params)
        # theta_0, theta_1, alpha, beta = params

        plot(x, y, x_star, params, range_min=range_min, range_max=range_max,
             plot_directory=args.plot_directory,
             target_name=current_target, param_name=args.x_column)


if __name__ == '__main__':
    run(parse_args())
