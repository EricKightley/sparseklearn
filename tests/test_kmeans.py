import unittest
import numpy as np
from sparseklearn import Sparsifier, KMeans

from generate_test_data import DataGenerator

class TestSparsifier(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        kmeans = KMeans(n_clusters = 2, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        self.kmeans = kmeans

    def test_fit_sparsifier(self):
        kmeans = KMeans(n_clusters = 2, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        kmeans.fit_sparsifier(X = self.td.X)
        self.assertArrayEqual(self.td.RHDX, kmeans.RHDX)
        self.assertArrayEqual(self.td.HDX, kmeans.HDX)
        self.assertArrayEqual(self.td.X, kmeans.X)
        self.assertArrayEqual(self.td.mask, kmeans.mask)
        self.assertEqual(self.td.N, kmeans.num_samp)
        self.assertEqual(self.td.Q, kmeans.num_feat_comp)
        self.assertEqual(self.td.P, kmeans.num_feat_full)

    def test_fit(self):
        self.kmeans.fit(X = self.td.X)




if __name__ == '__main__':
    unittest.main()

