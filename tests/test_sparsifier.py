import unittest
import numpy as np
from sparseklearn import Sparsifier

from generate_test_data import DataGenerator

class TestSparsifier(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        #spa = Sparsifier(mask = self.td.mask, data_dim = 5, transform = None)
        #spa.fit_sparsifier(HDX = self.td.HDX)
        #self.sparsifier = spa

    def test__generate_D_indices(self):
        spa = Sparsifier(num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                         num_samp = 4, transform = 'dct', D_indices = self.td.D_indices, 
                         mask = self.td.mask)
        np.random.seed(0)
        result = spa._generate_D_indices()
        correct = [1,2,4]
        self.assertArrayEqual(result, correct)

    def test_apply_HD(self):
        spa = Sparsifier(num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                         num_samp = 4, transform = 'dct', D_indices = self.td.D_indices, 
                         mask = self.td.mask)
        result = spa.apply_HD(self.td.X)
        self.assertArrayEqual(self.td.HDX, result)

    def test_invert_HD(self):
        spa = Sparsifier(num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                         num_samp = 4, transform = 'dct', D_indices = self.td.D_indices, 
                         mask = self.td.mask)
        result = spa.invert_HD(self.td.HDX)
        self.assertArrayEqual(self.td.X, result)

    def test_fit_all_passed(self):
        spa = Sparsifier(num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                         num_samp = 4, transform = 'dct', D_indices = self.td.D_indices, 
                         mask = self.td.mask)
        spa.fit_sparsifier(X=self.td.X, HDX = self.td.HDX, RHDX = self.td.RHDX)
        self.assertArrayEqual(self.td.X, spa.X)
        self.assertArrayEqual(self.td.HDX, spa.HDX)
        self.assertArrayEqual(self.td.RHDX, spa.RHDX)
        
    def test_fit_just_X_passed(self):
        spa = Sparsifier(num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                         num_samp = 4, transform = 'dct', D_indices = self.td.D_indices, 
                         mask = self.td.mask)
        spa.fit_sparsifier(X=self.td.X)
        self.assertArrayEqual(self.td.X, spa.X)
        self.assertArrayEqual(self.td.HDX, spa.HDX)
        self.assertArrayEqual(self.td.RHDX, spa.RHDX)


    """
    def test_fit(self):
        self.assertTrue(np.allclose(self.td.RHDX, self.sparsifier.RHDX, rtol=1e-6))
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
        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTHDX) / \
                                     np.dot(self.td.W.T, (self.td.RRTHDX!=0).astype(int))
        self.assertArrayEqual(first_moment_array, correct_first_moment_array)

    def test_weighted_means_and_variances(self):
        means,variances = self.sparsifier.weighted_means_and_variances(self.td.W)

        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTHDX) / \
                                     np.dot(self.td.W.T, (self.td.RRTHDX!=0).astype(int))
        correct_second_moment_array = np.dot(self.td.W.T, self.td.RRTHDX**2) / \
                                      np.dot(self.td.W.T, (self.td.RRTHDX!=0).astype(int))
        correct_variances = correct_second_moment_array - correct_first_moment_array**2
        self.assertArrayEqual(means, correct_first_moment_array)
        self.assertArrayEqual(variances, correct_variances)

    def test_pairwise_mahalanobis_distances(self):
        result_spherical = self.sparsifier.pairwise_mahalanobis_distances(\
            self.td.U, self.td.spherical_covariances, 'spherical') 
        result_diagonal = self.sparsifier.pairwise_mahalanobis_distances(\
            self.td.U, self.td.diagonal_covariances, 'diag')
        correct_spherical = self.td.correct_pairwise_mahalanobis_distances_spherical
        correct_diagonal = self.td.correct_pairwise_mahalanobis_distances_diagonal
        self.assertArrayEqual(result_spherical, correct_spherical)
        self.assertArrayEqual(result_diagonal, correct_diagonal)
        """

if __name__ == '__main__':
    unittest.main()




