import numpy as np


def pack_params(L, sigma_l_2, Theta_x):
    if not np.allclose(L, np.tril(L)):
        raise ValueError('Matrix L not lower triangular')

    params = L[np.tril_indices(len(L))]
    params = np.append(params, sigma_l_2)
    params = np.append(params, Theta_x)
    return params


def unpack_params(parameters_array, M):
    # M = num_tasks, we are required to know this
    expected_size_L = int(M * (M + 1) / 2)
    expected_size = expected_size_L + M + 2
    if len(parameters_array) != expected_size:
        raise ValueError('Wrong parameter size array. Expected %d got %d' % (expected_size, len(parameters_array)))
    L = np.zeros((M, M))
    L[np.tril_indices(M)] = parameters_array[0:expected_size_L]
    sigma_l_2 = parameters_array[expected_size_L:-2]
    Theta_x = [parameters_array[-2], parameters_array[-1]]
    return L, sigma_l_2, Theta_x
