import multitask
import numpy as np
import unittest


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        self.M = 5
        self.theta_size = 2

    def test_pack_unpack(self):
        L_orig = np.eye(self.M)
        sigma_l_2_orig = np.random.rand(self.M)
        theta_orig = np.random.rand(self.theta_size)

        for L_cur in [None, L_orig]:
            for sigma_cur in [None, sigma_l_2_orig]:
                for theta_cur in [None, theta_orig]:
                    incl_L = L_cur is not None
                    incl_sigma = sigma_cur is not None
                    incl_theta = theta_cur is not None
                    packed = multitask.utils.pack_params(L_cur, theta_cur, sigma_cur)
                    L_unp, theta_unp, sigma_unp = multitask.utils.unpack_params(packed, self.M,
                                                                                include_L=incl_L,
                                                                                include_sigma=incl_sigma,
                                                                                include_theta=incl_theta)

                    np.testing.assert_array_equal(L_cur, L_unp)
                    np.testing.assert_array_equal(sigma_cur, sigma_unp)
                    np.testing.assert_array_equal(theta_cur, theta_unp)

    def test_unpack_pack(self):
        for incl_L in [False, True]:
            for incl_sigma in [False, True]:
                for incl_theta in [False, True]:
                    arr_size = 0
                    if incl_L:
                        arr_size += int(self.M*(self.M+1)/2)
                    if incl_sigma:
                        arr_size += self.M
                    if incl_theta:
                        arr_size += self.theta_size

                    orig = np.random.rand(arr_size)
                    L_unp, theta_unp, sigma_unp = multitask.utils.unpack_params(orig, self.M,
                                                                                include_L=incl_L,
                                                                                include_sigma=incl_sigma,
                                                                                include_theta=incl_theta)

                    packed = multitask.utils.pack_params(L_unp, theta_unp, sigma_unp)
                    np.testing.assert_array_equal(orig, packed)

    def test_pack_nontriangular(self):
        K_f_orig = np.random.rand(self.M, self.M)
        sigma_l_2 = np.random.rand(self.M)
        theta_orig = np.random.rand(self.theta_size)

        with self.assertRaises(ValueError):
            multitask.utils.pack_params(K_f_orig, theta_orig, sigma_l_2)
