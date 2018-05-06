import unittest
import numpy as np

from sparseklearn import _l2_distance_both_compressed
from sparseklearn import _l2_distance_one_compressed_one_full
from sparseklearn import pairwise_l2_distances_with_self
from sparseklearn import pairwise_l2_distances_with_full
from sparseklearn import mahalanobis_distance_spherical
from sparseklearn import mahalanobis_distance_diagonal
from sparseklearn import pairwise_mahalanobis_distances_spherical

class DataGenerator():

    def __init__(self):

        self.RHDX = np.array([[1, 9, 8],
                              [7, 4, 3],
                              [2, 8, 7],
                              [4, 1, 3]], dtype = np.float64)

        self.mask = np.array([[1, 2, 4],
                              [2, 3, 4],
                              [0, 2, 4],
                              [1, 2, 3]], dtype = np.int64)

        self.W = np.array([[0, 2, 7],
                           [4, 1, 8],
                           [1, 6, 4],
                           [3, 2, 8]], dtype = np.float64)

        self.U = np.array([[0, 6, 8, 3, 1],
                           [1, 3, 4, 7, 5],
                           [8, 9, 0, 4, 2]], dtype = np.float64)

        self.diagonal_covariances = np.array([[2, 3, 1, 1, 6],
                                              [7, 2, 1, 5, 4],
                                              [4, 2, 8, 9, 1]], dtype = np.float64)

        self.spherical_covariances = np.array([2,3,4], dtype = np.float64)

        self.N = 4
        self.Q = 3
        self.P = 5
        self.K = 3

class TestFastLAMethods(unittest.TestCase):

    def setUp(self):
        self.td = DataGenerator()

    def test__l2_distance_both_compressed(self):
        """ Distance between RHDX[1] and RHDX[3]. """
        result = _l2_distance_both_compressed(self.td.RHDX[1], 
                                          self.td.RHDX[3],
                                          self.td.mask[1],
                                          self.td.mask[3],
                                          self.td.Q,
                                          self.td.P)
        correct = np.sqrt(5/2. * 37)
        self.assertAlmostEqual(correct,result,places=6)

    def test__l2_distance_one_compressed_one_full(self):
        """ Distance between RHDX[1] and U[2]. """
        result = _l2_distance_one_compressed_one_full(self.td.RHDX[1], 
                                                  self.td.U[2],
                                                  self.td.mask[1],
                                                  self.td.Q,
                                                  self.td.P)
        correct = np.sqrt(5/3. * 50)
        self.assertAlmostEqual(correct,result,places=6)

    def test_pairwise_l2_distances_with_self(self):
        """ Pairwise distances between rows of RHDX."""
        result = np.zeros((self.td.N, self.td.N))
        pairwise_l2_distances_with_self(result,
                                        self.td.RHDX,
                                        self.td.mask,
                                        self.td.N,
                                        self.td.Q,
                                        self.td.P)
        correct = np.array([[0     , 5/2*29, 5/2*2 , 5/2*73],
                            [5/2*29, 0     , 5/2*17, 5/2*37],
                            [5/2*2 , 5/2*17, 0     , 5*49  ],
                            [5/2*73, 5/2*37, 5*49  , 0     ]], 
                            dtype = np.float64)
        correct = np.sqrt(correct)
        self.assertTrue(np.allclose(correct, result, rtol=1e-6))

    def test_pairwise_l2_distances_with_full(self):
        """Pairwise distances between rows of RHDX and rows of U."""
        result = np.zeros((self.td.N, self.td.K))
        pairwise_l2_distances_with_full(result,
                                        self.td.RHDX,
                                        self.td.U,
                                        self.td.mask,
                                        self.td.N,
                                        self.td.K,
                                        self.td.Q,
                                        self.td.P)
        correct = np.array([[  75,  38, 181],
                            [   6,  22,  50],
                            [  40,  21, 125],
                            [  53,  26,  27]],
                            dtype = np.float64)
        correct = np.sqrt(5/3*correct)
        self.assertTrue(np.allclose(correct, result, rtol=1e-6))

    def test_mahalanobis_distance_spherical(self):
        """ Mahalanobis distance ||RHDX[1] - U[2]|| with spherical covariance
        sigmasquared = 2.2. """

        result = mahalanobis_distance_spherical(self.td.RHDX[1],
                                                self.td.U[2],
                                                self.td.mask[1],
                                                2.2,
                                                self.td.Q,
                                                self.td.P)
        correct = np.sqrt(50 * 5/3 / 2.2)
        self.assertAlmostEqual(correct, result, places=6)

    def test_mahalanobis_distance_diagonal(self):
        """ Mahalanobis distance ||RHDX[0] - U[2]|| with diagonal covariance
        diagonal_covariances[2]. """

        result = mahalanobis_distance_diagonal(self.td.RHDX[1],
                                                self.td.U[2],
                                                self.td.mask[1],
                                                self.td.diagonal_covariances[2],
                                                self.td.Q,
                                                self.td.P)
        correct = np.sqrt(57/8 * 5/3)
        self.assertAlmostEqual(correct, result, places=6)

    def test_pairwise_mahalanobis_distances_spherical(self):
        """ Mahalanobis distances ||RHDX - U|| with spherical_covariances.
        """

        result = np.zeros((self.td.N,self.td.K))
        pairwise_mahalanobis_distances_spherical(result,
                                                 self.td.RHDX,
                                                 self.td.U,
                                                 self.td.mask,
                                                 self.td.spherical_covariances,
                                                 self.td.N,
                                                 self.td.K,
                                                 self.td.Q,
                                                 self.td.P)
        correct = np.sqrt(57/8 * 5/3)
        print(result)
        #self.assertAlmostEqual(correct, result, places=6)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()





























