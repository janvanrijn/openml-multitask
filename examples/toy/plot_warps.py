import argparse
import arff
import functools
import matplotlib.pyplot as plt
import multitask
import numpy as np
import os
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept input warping')
    parser.add_argument('--plot_directory', type=str, default=os.path.expanduser("~") + '/experiments/multitask/warped/')
    parser.add_argument('--default_value_alpha', type=float, default=100)
    parser.add_argument('--default_value_beta', type=float, default=1)
    parser.add_argument('--exp_decay_base', type=float, default=0.5)
    parser.add_argument('--exp_decay_exp', type=float, default=10)
    parser.add_argument('--min_value', type=float, default=-10)
    parser.add_argument('--max_value', type=float, default=10)

    return parser.parse_args()


def run():
    args = parse_args()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    f.suptitle('Sharing x per column, y per row')

    input_values = np.linspace(args.min_value, args.max_value, 100)
    exp_decay = lambda x: args.exp_decay_base ** (x * args.exp_decay_exp)
    warping_fn = lambda x: scipy.stats.beta.pdf(x, a=args.default_value_alpha,
                                                b=args.default_value_beta,
                                                loc=args.min_value,
                                                scale=(args.max_value-args.min_value))
    ax1.plot(input_values, exp_decay(input_values))
    ax2.plot(input_values, warping_fn(input_values))
    ax3.plot(input_values, exp_decay(warping_fn(input_values)))
    plt.show()


if __name__ == '__main__':
    run()
