import arff
import multitask
import numpy as np
import unittest


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        data_filepath = '../../data/svm-gamma-10tasks.arff'
        with open(data_filepath, 'r') as fp:
            dataset = arff.load(fp)

        data = np.array(dataset['data'])
        self.x = data[:, 0]
        self.y = data[:, 1]

        # this checks if the dataset that we work on still is the same
        self.assertEqual(dataset['attributes'][0][0], 'gamma-log')
        self.assertEqual(dataset['attributes'][1][0], 'score-at-task-3')

        self.x_star = np.linspace(min(self.x), max(self.x), 4 * len(self.x))

    def test_get_posterior(self):
        mu, sigma, lmk = multitask.utils.get_posterior(self.x_star, self.x, self.y)

        mu_prime = []
        sigma_prime = []
        lmk_prime = None
        for x_star_single in list(self.x_star):
            result = multitask.utils.get_posterior_single_point(x_star_single, self.x, self.y)
            mu_prime.append(result[0])
            sigma_prime.append(result[1])
            lmk_prime = result[2]

        np.testing.assert_array_almost_equal(mu, np.array(mu_prime))
        # sigma contains covariance matrix. sigma prime contains variances. Diagonal of cov matrix = variances
        np.testing.assert_array_almost_equal(np.diagonal(sigma), np.array(sigma_prime))
        self.assertEqual(lmk, lmk_prime)
