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
        kmeans.fit_sparsifier(X = self.td.X)
        self.kmeans = kmeans

    def test_fit_sparsifier(self):
        self.assertArrayEqual(self.td.RHDX, self.kmeans.RHDX)

if __name__ == '__main__':
    unittest.main()

