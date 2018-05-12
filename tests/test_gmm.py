import unittest
import numpy as np
from sparseklearn import GaussianMixture

from generate_test_data import DataGenerator

class TestGaussianMixture(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        #self.gmm = GaussianMixture()
        #spa = Sparsifier(mask = self.td.mask, data_dim = 5, transform = None)
        #spa.fit_sparsifier(X = self.td.X)
        #self.sparsifier = spa

    def test_fit_sparsifier(self):
        gmm = GaussianMixture(n_components = 2, mask = self.td.mask, 
                data_dim = 5, transform = None)
        gmm.fit_sparsifier(X = self.td.X)
        self.assertTrue(np.allclose(self.td.RX, gmm.RHDX, rtol=1e-6))
        self.assertTrue(np.allclose(self.td.mask, gmm.mask, rtol=1e-6))
        self.assertEqual(self.td.N, gmm.N)
        self.assertEqual(self.td.Q, gmm.Q)
        self.assertEqual(self.td.P, gmm.P)


