import numpy as np


def pack_params(L, Theta_x=None, sigma_l_2=None):
    if not np.allclose(L, np.tril(L)):
        raise ValueError('Matrix L not lower triangular')

    params = L[np.tril_indices(len(L))]
    if Theta_x is not None:
        params = np.append(params, Theta_x)
    if sigma_l_2 is not None:
        params = np.append(params, sigma_l_2)
    return params


def unpack_params(parameters_array, M, include_theta=True, include_sigma=True):
    # M = num_tasks, we are required to know this
    expected_size_L = int(M * (M + 1) / 2)
    expected_size = expected_size_L
    if include_theta:
        expected_size += 2
    if include_sigma:
        expected_size += M

    if len(parameters_array) != expected_size:
        raise ValueError('Wrong parameter size array. Expected %d got %d' % (expected_size, len(parameters_array)))
    L = np.zeros((M, M))
    L[np.tril_indices(M)] = parameters_array[0:expected_size_L]
    if include_theta:
        Theta_x = [parameters_array[expected_size_L], parameters_array[expected_size_L+1]]
    else:
        Theta_x = [1, 1] # TODO: defaults

    if include_sigma:
        sigma_l_2 = parameters_array[len(parameters_array)-M:]
    else:
        sigma_l_2 = np.zeros(M)
    return L, Theta_x, sigma_l_2
