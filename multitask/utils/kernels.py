import numpy as np


def rbf_kernel1D(x_a, x_b, theta0=1, theta1=1):
    # for 1 dimensional data points. expects x_a and x_b to be a list of floats
    return theta0 * np.exp(-0.5 * theta1 * np.subtract.outer(x_a, x_b) ** 2)