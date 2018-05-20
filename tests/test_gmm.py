import unittest
import numpy as np
from sparseklearn import GaussianMixture

from generate_test_data import DataGenerator

class TestGaussianMixture(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        gmm = GaussianMixture(n_components = 3, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        self.gmm = gmm

    def test_fit_sparsifier(self):
        gmm = GaussianMixture(n_components = 3, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        gmm.fit_sparsifier(X = self.td.X)
        self.assertArrayEqual(self.td.RHDX, gmm.RHDX)
        self.assertArrayEqual(self.td.HDX, gmm.HDX)
        self.assertArrayEqual(self.td.X, gmm.X)
        self.assertArrayEqual(self.td.mask, gmm.mask)
        self.assertEqual(self.td.N, gmm.num_samp)
        self.assertEqual(self.td.Q, gmm.num_feat_comp)
        self.assertEqual(self.td.P, gmm.num_feat_full)

    def test__compute_logdet_array(self):
        gmm = GaussianMixture(n_components = 3, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        logdet_spherical = gmm._compute_logdet_array(self.td.spherical_covariances, 'spherical')
        logdet_diag = gmm._compute_logdet_array(self.td.diagonal_covariances, 'diag')
        self.assertArrayEqual(self.td.correct_logdet_spherical, logdet_spherical)
        self.assertArrayEqual(self.td.correct_logdet_diag, logdet_diag)

    def test__compute_log_prob(self):
        #TODO
        return 1
        
    def test__compute_log_resp(self):
        #TODO
        return 1

    def test__estimate_gaussian_means_and_covariances(self):
        X = np.array([[0,1,0,0],
                      [1,0,0,0],
                      [0,0,1,2]], dtype = np.float64)
        gmm = GaussianMixture(n_components = 2, num_feat_full = 4, num_feat_comp = 4,
                              num_feat_shared = 4, num_samp = 3, transform = None, 
                              D_indices = None, mask = None, reg_covar = 0)
        gmm.fit_sparsifier(X = X)
        # note: columns should sum to 1, but don't have to because the weighted
        # means computation has to renormalize anyway to account for the mask
        resp = np.array([[.6, .3],
                         [.4, .2],
                         [ 0, .5]], dtype = np.float64)
        means, covariances = gmm._estimate_gaussian_means_and_covariances(resp, 'diag')
        correct_means = np.array([[ .4, .6, .0, .0],
                                  [ .2, .3, .5, 1.]], dtype=np.float64)
        correct_covariances = np.array([[.24, .24,   0,  0],
                                        [.16, .21, .25,  1]], dtype=np.float64)
        self.assertArrayEqual(correct_means, means)
        self.assertArrayEqual(correct_covariances, covariances)

    def test__estimate_gaussian_weights(self):
        #TODO
        return 1

    def test_fit(self):
        """ Catches a case where covariance goes to 0."""
        reg_covar = 1e-6
        np.random.seed(0)
        X = np.array([[0,1,0,0],
                      [1,0,0,0],
                      [0,0,1,2]], dtype = np.float64)
        gmm = GaussianMixture(n_components = 2, covariance_type = 'diag',
                num_feat_full = 4, num_feat_comp = 2,
                num_feat_shared = 1, num_samp = 3, transform = None, D_indices = None,
                mask = None, reg_covar = reg_covar, init_params = 'random', max_iter = 5)
        gmm.fit(X)
        correct_means = np.array([[0, 0, 1, 0],
                                  [1, 0, 0, 0]], dtype = np.float64)
        correct_covariances = np.ones_like(correct_means)*reg_covar
        self.assertArrayEqual(gmm.means_, correct_means)
        self.assertArrayEqual(gmm.covariances_, correct_covariances)




    #def test_fit(self):
        #self.gmm.fit(X = self.td.X)

if __name__ == '__main__':
    unittest.main()

