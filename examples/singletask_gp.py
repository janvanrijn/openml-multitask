import argparse
import arff
import matplotlib.pyplot as plt
import multitask
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--data_file', type=str, default='../data/svm-gamma-10tasks.arff')
    parser.add_argument('--x_column', type=str, default='gamma-log')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--plot_directory', type=str, default='C:/experiments/multitask/single/')

    return parser.parse_args()


def plot(x_train, y_train, x, mu, sigma, target_name, param_name, num_samples=3):
    fig, ax = plt.subplots()

    variance = sigma.diagonal()
    ax.fill_between(x, mu - variance ** 0.5, mu + variance ** 0.5, color="#dddddd")
    ax.plot(x, mu, 'r--', lw=2)

    if num_samples:
        for i in range(num_samples):
            ax.plot(x, np.random.multivariate_normal(mu, sigma))

    ax.plot(x_train, y_train, 'bs', ms=4)
    ax.set(xlabel=param_name, ylabel='predictive_accuracy',
           title='Single Task GP on ' + target_name)

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0., 1.])
    fig.savefig(fname=target_name)


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
        print(current_target)
        y = data[:, y_idx]
        mu, sigma, _ = multitask.utils.get_posterior(x_star, x, y)
        plot(x, y, x_star, mu, sigma, target_name=args.plot_directory + current_target, param_name=args.x_column)


if __name__ == '__main__':
    run(parse_args())
