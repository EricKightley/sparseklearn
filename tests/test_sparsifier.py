import unittest
import numpy as np
from sparseklearn import Sparsifier

from generate_test_data import DataGenerator

class TestSparsifier(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        spa = Sparsifier(mask = self.td.mask, data_dim = 5, transform = None)
        spa.fit_sparsifier(X = self.td.X)
        self.sparsifier = spa

    def test_fit(self):
        self.assertTrue(np.allclose(self.td.RX, self.sparsifier.RHDX, rtol=1e-6))
        self.assertTrue(np.allclose(self.td.mask, self.sparsifier.mask, rtol=1e-6))
        self.assertEqual(self.td.N, self.sparsifier.N)
        self.assertEqual(self.td.Q, self.sparsifier.Q)
        self.assertEqual(self.td.P, self.sparsifier.P)

    def test_pairwise_distances(self):
        result_self = self.sparsifier.pairwise_distances()
        correct_self = self.td.correct_pairwise_l2_distances_with_self

        result_full = self.sparsifier.pairwise_distances(Y=self.td.U)
        correct_full = self.td.correct_pairwise_l2_distances_with_full

        self.assertArrayEqual(result_self, correct_self)
        #self.assertAlmostEqual(correct,result,places=6)

    def test_weighted_means(self):
        first_moment_array = self.sparsifier.weighted_means(self.td.W)
        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTX) / \
                                     np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        self.assertArrayEqual(first_moment_array, correct_first_moment_array)

    def test_weighted_means_and_variances(self):
        means,variances = self.sparsifier.weighted_means_and_variances(self.td.W)

        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTX) / \
                                     np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        correct_second_moment_array = np.dot(self.td.W.T, self.td.RRTX**2) / \
                                      np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        correct_variances = correct_second_moment_array - correct_first_moment_array**2
        self.assertArrayEqual(means, correct_first_moment_array)
        self.assertArrayEqual(variances, correct_variances)

if __name__ == '__main__':
    unittest.main()




