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

if __name__ == '__main__':
    unittest.main()




"""
import numpy as np
import h5py

from sparseklearn import Sparsifier
from sparseklearn import KMeans
from auxutils import generate_mnist_dataset

from sklearn.random_projection import GaussianRandomProjection as grp
from sklearn.random_projection import johnson_lindenstrauss_min_dim as jlmindim

# set the random seed
rs = 22
np.random.seed(rs)

# load computed data
hdf5_file = h5py.File('/home/eric/kmeansdata/sample_mnist.hdf5','r')
X = hdf5_file['X'][:]
HDX = hdf5_file['HDX'][:]
#RHDX = hdf5_file['RHDX'][:]
#mask = hdf5_file['mask'][:]
#precond_D = hdf5_file['precond_D'][:]
#labels = hdf5_file['labels'][:]

X = X[:1000]
N = X.shape[0]
P = 784

#Q = jlmindim(n_samples = N, eps = 0.5)
Q = 30

compression_target = float(Q/P)

spr = Sparsifier(compression_target = compression_target, alpha_target = 0.0, transform = 'dct')
spr.fit_sparsifier(X = X)
pd = spr.pairwise_distances()
pd2 = spr.pairwise_distances(U=HDX[2:4])


#from sklearn.metrics.pairwise import pairwise_distances
#pd_true = pairwise_distances(X)
#pd_grp = pairwise_distances(grp(n_components = Q).fit_transform(X))
#I = np.identity(N)
#normalizer = pd_true + np.identity(N)
#pd = (pd + I) / normalizer
#pd_grp = (pd_grp + I) / normalizer
"""




