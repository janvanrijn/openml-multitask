import numpy as np


def pack_params(L=None, Theta_x=None, sigma_l_2=None):
    if L is not None and not np.allclose(L, np.tril(L)):
        raise ValueError('Matrix L not lower triangular')

    params = list()
    if L is not None:
        params = L[np.tril_indices(len(L))]
    if Theta_x is not None:
        params = np.append(params, Theta_x)
    if sigma_l_2 is not None:
        params = np.append(params, sigma_l_2)
    return params


def unpack_params(parameters_array, M, include_L=True, include_theta=True, include_sigma=True):
    # M = num_tasks, we are required to know this
    expected_size_L = int(M * (M + 1) / 2)
    expected_size = 0
    if include_L:
        expected_size += expected_size_L
    if include_theta:
        expected_size += 2
    if include_sigma:
        expected_size += M

    if len(parameters_array) != expected_size:
        raise ValueError('Wrong parameter size array. Expected %d got %d' % (expected_size, len(parameters_array)))

    L = None
    Theta_x = None
    sigma_l_2 = None

    # L is always first entrees
    if include_L:
        L = np.zeros((M, M))
        L[np.tril_indices(M)] = parameters_array[0:expected_size_L]

        # in this case theta comes after L
        if include_theta:
            Theta_x = [parameters_array[expected_size_L], parameters_array[expected_size_L + 1]]
    elif include_theta:
        Theta_x = [parameters_array[0], parameters_array[1]]

    # sigma is always last entrees
    if include_sigma:
        sigma_l_2 = parameters_array[len(parameters_array)-M:]

    return L, Theta_x, sigma_l_2
