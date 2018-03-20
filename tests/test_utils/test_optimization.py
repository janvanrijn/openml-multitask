import multitask
import numpy as np
import unittest


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        self.M = 5
        self.theta_size = 2

    def test_pack_unpack(self):
        L_orig = np.eye(self.M)
        sigma_l_2 = np.random.rand(self.M)
        theta_orig = np.random.rand(self.theta_size)

        packed = multitask.utils.pack_params(L_orig, sigma_l_2, theta_orig)
        L_unpacked, sigma_l_2_unpacked, theta_unpacked = multitask.utils.unpack_params(packed, self.M)

        np.testing.assert_array_equal(L_orig, L_unpacked)
        np.testing.assert_array_equal(sigma_l_2, sigma_l_2_unpacked)
        np.testing.assert_array_equal(theta_orig, theta_unpacked)

    def test_unpack_pack(self):
        arr_size = int(self.M*(self.M+1)/2) + self.M + self.theta_size
        orig = np.random.rand(arr_size)
        L_unpacked, sigma_l_2_unpacked, theta_unpacked = multitask.utils.unpack_params(orig, self.M)

        packed = multitask.utils.pack_params(L_unpacked, sigma_l_2_unpacked, theta_unpacked)
        np.testing.assert_array_equal(packed, packed)

    def test_pack_triangular(self):
        K_f_orig = np.random.rand(self.M, self.M)
        sigma_l_2 = np.random.rand(self.M)
        theta_orig = np.random.rand(self.theta_size)

        with self.assertRaises(ValueError):
            multitask.utils.pack_params(K_f_orig, sigma_l_2, theta_orig)
