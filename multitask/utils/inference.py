import multitask
import numpy as np


def get_posterior(x_star, x, y):
    # implements Eq. 2.19 from Rasmussen and Williams
    if not isinstance(x_star, np.ndarray):
        raise ValueError()
    if not isinstance(x, np.ndarray):
        raise ValueError()
    if not isinstance(y, np.ndarray):
        raise ValueError()
    if not x.shape == y.shape:
        raise ValueError()
    if not (x_star.ndim == x.ndim == y.ndim == 1):
        raise ValueError()

    K_vv = multitask.utils.rbf_kernel1D(x, x)
    K_sv = multitask.utils.rbf_kernel1D(x_star, x)
    K_vs = multitask.utils.rbf_kernel1D(x, x_star)
    K_ss = multitask.utils.rbf_kernel1D(x_star, x_star)

    K_vv_inv = np.linalg.inv(K_vv)

    mu = K_sv.dot(K_vv_inv).dot(y)               # predictive means
    sigma = K_ss - K_sv.dot(K_vv_inv).dot(K_vs)  # covariance matrix (stdevs are on the diagonal)
    return mu, sigma


def get_posterior_single_point(x_star, x, y):
    # implements Eq. 2.25/2.26 from Rasmussen and Williams
    if not isinstance(x_star, float):
        raise ValueError()
    if not isinstance(x, np.ndarray):
        raise ValueError()
    if not isinstance(y, np.ndarray):
        raise ValueError()
    if not x.shape == y.shape:
        raise ValueError()
    if not (x.ndim == y.ndim == 1):
        raise ValueError()

    k_vv = multitask.utils.rbf_kernel1D(x, x)
    k_sv = multitask.utils.rbf_kernel1D(x_star, x)
    k_vs = multitask.utils.rbf_kernel1D(x, x_star)
    k_ss = multitask.utils.rbf_kernel1D(x_star, x_star)

    k_vv_inv = np.linalg.inv(k_vv)

    mu = k_sv.T.dot(k_vv_inv).dot(y)
    variance = k_ss - k_sv.T.dot(k_vv_inv).dot(k_sv)

    return mu, variance