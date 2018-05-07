import unittest
import numpy as np

from sparseklearn import _l2_distance_both_compressed
from sparseklearn import _l2_distance_one_compressed_one_full
from sparseklearn import pairwise_l2_distances_with_self
from sparseklearn import pairwise_l2_distances_with_full
from sparseklearn import mahalanobis_distance_spherical
from sparseklearn import mahalanobis_distance_diagonal
from sparseklearn import pairwise_mahalanobis_distances_spherical
from sparseklearn import pairwise_mahalanobis_distances_diagonal
from sparseklearn import update_first_moment_single_sample
from sparseklearn import update_both_moments_single_sample
from sparseklearn import update_first_moment_array_single_sample
from sparseklearn import update_both_moment_arrays_single_sample

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
        """ Mahalanobis distance ||RHDX[1] - U[2]|| with diagonal covariance
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
        correct = np.array([[ 75/2,  38/3, 181/4],
                            [  6/2,  22/3,  50/4],
                            [ 40/2,  21/3, 125/4],
                            [ 53/2,  26/3,  27/4]],
                            dtype = np.float64)
        correct = np.sqrt(5/3*correct)
        self.assertTrue(np.allclose(correct, result, rtol=1e-6))

    def test_pairwise_mahalanobis_distances_diagonal(self):
        """ Mahalanobis distances ||RHDX - U|| with diagonal_covariances.
        """

        result = np.zeros((self.td.N,self.td.K))
        pairwise_mahalanobis_distances_diagonal(result,
                                                self.td.RHDX,
                                                self.td.U,
                                                self.td.mask,
                                                self.td.diagonal_covariances,
                                                self.td.N,
                                                self.td.K,
                                                self.td.Q,
                                                self.td.P)
        correct = np.array([[   17.5,   29.25, 78.125],
                            [    8/3,    11.8,  7.125],
                            [      8,  17+1/7,     42],
                            [  151/3,    12.7, 12+53/72]],
                            dtype = np.float64)
        correct = np.sqrt(5/3*correct)
        self.assertTrue(np.allclose(correct, result, rtol=1e-6))

    def test_update_first_moment_single_sample(self):
        """ Update a (init to zero) weighted mean and normalizer using 
        X[1], W[1,0]. """

        first_moment_to_update = np.zeros(self.td.P)
        normalizer_to_update = np.zeros(self.td.P)
        update_first_moment_single_sample(first_moment_to_update,
                                          normalizer_to_update,
                                          self.td.RHDX[1],
                                          self.td.mask[1],
                                          self.td.W[1,0],
                                          self.td.Q,
                                          self.td.P)
        correct_moment = np.array([0, 0, 28, 16, 12], dtype = np.float64)
        correct_normalizer = np.array([0, 0, 4, 4, 4], dtype = np.float64)

        self.assertTrue(np.allclose(correct_moment, first_moment_to_update, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_normalizer, normalizer_to_update, 
            rtol = 1e-6))

    def test_update_both_moments_single_sample(self):
        """ Update a (init to zero) weighted mean and normalizer using 
        X[1], W[1,0]. """

        first_moment_to_update = np.zeros(self.td.P)
        second_moment_to_update = np.zeros(self.td.P)
        normalizer_to_update = np.zeros(self.td.P)
        update_both_moments_single_sample(first_moment_to_update,
                                          second_moment_to_update,
                                          normalizer_to_update,
                                          self.td.RHDX[1],
                                          self.td.mask[1],
                                          self.td.W[1,0],
                                          self.td.Q,
                                          self.td.P)
        correct_first_moment = np.array([0, 0, 28, 16, 12], dtype = np.float64)
        correct_second_moment = np.array([0, 0, 196, 64, 36])
        correct_normalizer = np.array([0, 0, 4, 4, 4], dtype = np.float64)

        self.assertTrue(np.allclose(correct_first_moment, first_moment_to_update, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_second_moment, second_moment_to_update, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_normalizer, normalizer_to_update, 
            rtol = 1e-6))

    def test_update_first_moment_array_single_sample(self):
        """ Update a set of 3 zero-initialized means using X[2], W[2,:]."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        normalizer_array = np.zeros((self.td.K, self.td.P))
        update_first_moment_array_single_sample(first_moment_array,
                                               normalizer_array,
                                               self.td.RHDX[2],
                                               self.td.mask[2],
                                               self.td.W[2,:],
                                               self.td.K,
                                               self.td.Q,
                                               self.td.P)
        correct_first_moment_array = np.array([[  2,  0,  8,  0,  7],
                                               [ 12,  0, 48,  0, 42],
                                               [  8,  0, 32,  0, 28]], 
                                               dtype = np.float64)
        correct_normalizer_array = np.array([[1,0,1,0,1],
                                            [6,0,6,0,6],
                                            [4,0,4,0,4]],
                                            dtype = np.float64)

        self.assertTrue(np.allclose(correct_first_moment_array, first_moment_array, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_normalizer_array, normalizer_array, 
            rtol = 1e-6))

    def test_update_both_moment_arrays_single_sample(self):
        """ Update a set of 3 zero-initialized means using X[2], W[2,:]."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        second_moment_array = np.zeros((self.td.K, self.td.P))
        normalizer_array = np.zeros((self.td.K, self.td.P))
        update_both_moment_arrays_single_sample(first_moment_array,
                                               second_moment_array,
                                               normalizer_array,
                                               self.td.RHDX[2],
                                               self.td.mask[2],
                                               self.td.W[2,:],
                                               self.td.K,
                                               self.td.Q,
                                               self.td.P)
        correct_first_moment_array = np.array([[  2,  0,  8,  0,  7],
                                               [ 12,  0, 48,  0, 42],
                                               [  8,  0, 32,  0, 28]], 
                                               dtype = np.float64)
        correct_second_moment_array = np.array([[  4,  0,  64,  0,  49],
                                               [ 24,  0, 384,  0, 294],
                                               [  16,  0, 256,  0, 196]], 
                                               dtype = np.float64)
        correct_normalizer_array = np.array([[1,0,1,0,1],
                                            [6,0,6,0,6],
                                            [4,0,4,0,4]],
                                            dtype = np.float64)

        self.assertTrue(np.allclose(correct_first_moment_array, first_moment_array, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_second_moment_array, second_moment_array, 
            rtol = 1e-6))
        self.assertTrue(np.allclose(correct_normalizer_array, normalizer_array, 
            rtol = 1e-6))

if __name__ == '__main__':
    unittest.main()





























