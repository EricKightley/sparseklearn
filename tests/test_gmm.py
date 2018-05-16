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

        


    #def test_fit(self):
        #self.gmm.fit(X = self.td.X)



