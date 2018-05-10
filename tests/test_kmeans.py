import unittest
import numpy as np
from sparseklearn import Sparsifier, KMeans

from generate_test_data import DataGenerator

class TestSparsifier(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        kmeans = KMeans(mask = self.td.mask, data_dim = 5, transform = None)
        kmeans.fit_sparsifier(X = self.td.X)
        self.kmeans = kmeans

    def test_fit(self):
        self.assertTrue(np.allclose(self.td.RX, self.kmeans.RHDX, rtol=1e-6))
        self.assertTrue(np.allclose(self.td.mask, self.kmeans.mask, rtol=1e-6))
        self.assertEqual(self.td.N, self.kmeans.N)
        self.assertEqual(self.td.Q, self.kmeans.Q)
        self.assertEqual(self.td.P, self.kmeans.P)

if __name__ == '__main__':
    unittest.main()

