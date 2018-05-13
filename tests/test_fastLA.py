import unittest
import numpy as np

from sparseklearn import dist_both_comp
from sparseklearn import dist_one_comp_one_full
from sparseklearn import pairwise_l2_distances_with_self
from sparseklearn import pairwise_l2_distances_with_full
from sparseklearn import mahalanobis_distance_spherical
from sparseklearn import mahalanobis_distance_diagonal
from sparseklearn import pairwise_mahalanobis_distances_spherical
from sparseklearn import pairwise_mahalanobis_distances_diagonal

from sparseklearn import update_weighted_first_moment
from sparseklearn import update_weighted_first_moment_array
from sparseklearn import compute_weighted_first_moment_array
from sparseklearn import update_weighted_first_and_second_moment
from sparseklearn import update_weighted_first_and_second_moment_array
from sparseklearn import compute_weighted_first_and_second_moment_array

from sparseklearn import apply_mask_to_full_sample

from generate_test_data import DataGenerator

class TestFastLAMethods(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()

    def test_dist_both_comp(self):
        """ Distance between RX[1] and RX[3]. """
        result = dist_both_comp(self.td.RX[1], 
                                          self.td.RX[3],
                                          self.td.mask[1],
                                          self.td.mask[3],
                                          self.td.Q,
                                          self.td.P)
        correct = np.sqrt(5/2. * 37)
        self.assertAlmostEqual(correct,result,places=6)

    def test_dist_one_comp_one_full(self):
        """ Distance between RX[1] and U[2]. """
        result = dist_one_comp_one_full(self.td.RX[1], 
                                                  self.td.U[2],
                                                  self.td.mask[1],
                                                  self.td.Q,
                                                  self.td.P)
        correct = np.sqrt(5/3. * 50)
        self.assertAlmostEqual(correct,result,places=6)

    def test_pairwise_l2_distances_with_self(self):
        """ Pairwise distances between rows of RX."""
        result = np.zeros((self.td.N, self.td.N))
        pairwise_l2_distances_with_self(result,
                                        self.td.RX,
                                        self.td.mask,
                                        self.td.N,
                                        self.td.Q,
                                        self.td.P)
        correct = self.td.correct_pairwise_l2_distances_with_self
        self.assertArrayEqual(correct, result)

    def test_pairwise_l2_distances_with_full(self):
        """Pairwise distances between rows of RX and rows of U."""
        result = np.zeros((self.td.N, self.td.K))
        pairwise_l2_distances_with_full(result,
                                        self.td.RX,
                                        self.td.U,
                                        self.td.mask,
                                        self.td.N,
                                        self.td.K,
                                        self.td.Q,
                                        self.td.P)
        correct = self.td.correct_pairwise_l2_distances_with_full
        self.assertArrayEqual(correct, result)

    def test_mahalanobis_distance_spherical(self):
        """ Mahalanobis distance ||RX[1] - U[2]|| with spherical covariance
        sigmasquared = 2.2. """

        result = mahalanobis_distance_spherical(self.td.RX[1],
                                                self.td.U[2],
                                                self.td.mask[1],
                                                2.2,
                                                self.td.Q,
                                                self.td.P)
        correct = np.sqrt(50 * 5/3 / 2.2)
        self.assertAlmostEqual(correct, result, places=6)

    def test_mahalanobis_distance_diagonal(self):
        """ Mahalanobis distance ||RX[1] - U[2]|| with diagonal covariance
        diagonal_covariances[2]. """

        result = mahalanobis_distance_diagonal(self.td.RX[1],
                                                self.td.U[2],
                                                self.td.mask[1],
                                                self.td.diagonal_covariances[2],
                                                self.td.Q,
                                                self.td.P)
        correct = np.sqrt(57/8 * 5/3)
        self.assertAlmostEqual(correct, result, places=6)

    def test_pairwise_mahalanobis_distances_spherical(self):
        """ Mahalanobis distances ||RX - U|| with spherical_covariances.
        """

        result = np.zeros((self.td.N,self.td.K))
        pairwise_mahalanobis_distances_spherical(result,
                                                 self.td.RX,
                                                 self.td.U,
                                                 self.td.mask,
                                                 self.td.spherical_covariances,
                                                 self.td.N,
                                                 self.td.K,
                                                 self.td.Q,
                                                 self.td.P)
        correct = self.td.correct_pairwise_mahalanobis_distances_spherical
        self.assertArrayEqual(correct, result)

    def test_pairwise_mahalanobis_distances_diagonal(self):
        """ Mahalanobis distances ||RX - U|| with diagonal_covariances.
        """

        result = np.zeros((self.td.N,self.td.K))
        pairwise_mahalanobis_distances_diagonal(result,
                                                self.td.RX,
                                                self.td.U,
                                                self.td.mask,
                                                self.td.diagonal_covariances,
                                                self.td.N,
                                                self.td.K,
                                                self.td.Q,
                                                self.td.P)
        correct = self.td.correct_pairwise_mahalanobis_distances_diagonal
        self.assertArrayEqual(correct, result)

    def test_update_weighted_first_moment(self):
        """ Update a (init to zero) weighted mean and normalizer using 
        X[1], W[1,0]. """

        first_moment_to_update = np.zeros(self.td.P)
        normalizer_to_update = np.zeros(self.td.P)
        update_weighted_first_moment(first_moment_to_update,
                                          normalizer_to_update,
                                          self.td.RX[1],
                                          self.td.mask[1],
                                          self.td.W[1,0],
                                          self.td.Q,
                                          self.td.P)
        correct_moment = np.array([0, 0, 28, 16, 12], dtype = np.float64)
        correct_normalizer = np.array([0, 0, 4, 4, 4], dtype = np.float64)

        self.assertArrayEqual(correct_moment, first_moment_to_update)
        self.assertArrayEqual(correct_normalizer, normalizer_to_update)

    def test_update_weighted_first_and_second_moment(self):
        """ Update a (init to zero) weighted mean and normalizer using 
        X[1], W[1,0]. """

        first_moment_to_update = np.zeros(self.td.P)
        second_moment_to_update = np.zeros(self.td.P)
        normalizer_to_update = np.zeros(self.td.P)
        update_weighted_first_and_second_moment(first_moment_to_update,
                                          second_moment_to_update,
                                          normalizer_to_update,
                                          self.td.RX[1],
                                          self.td.mask[1],
                                          self.td.W[1,0],
                                          self.td.Q,
                                          self.td.P)
        correct_first_moment = np.array([0, 0, 28, 16, 12], dtype = np.float64)
        correct_second_moment = np.array([0, 0, 196, 64, 36])
        correct_normalizer = np.array([0, 0, 4, 4, 4], dtype = np.float64)

        self.assertArrayEqual(correct_first_moment, first_moment_to_update)
        self.assertArrayEqual(correct_second_moment, second_moment_to_update)
        self.assertArrayEqual(correct_normalizer, normalizer_to_update)

    def test_update_weighted_first_moment_array(self):
        """ Update a set of 3 zero-initialized means using X[2], W[2,:]."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        normalizer_array = np.zeros((self.td.K, self.td.P))
        update_weighted_first_moment_array(first_moment_array,
                                               normalizer_array,
                                               self.td.RX[2],
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

        self.assertArrayEqual(correct_first_moment_array, first_moment_array)
        self.assertArrayEqual(correct_normalizer_array, normalizer_array)

    def test_update_weighted_first_and_second_moment_array(self):
        """ Update a set of 3 zero-initialized means using X[2], W[2,:]."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        second_moment_array = np.zeros((self.td.K, self.td.P))
        normalizer_array = np.zeros((self.td.K, self.td.P))
        update_weighted_first_and_second_moment_array(first_moment_array,
                                               second_moment_array,
                                               normalizer_array,
                                               self.td.RX[2],
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

        self.assertArrayEqual(correct_first_moment_array, first_moment_array)
        self.assertArrayEqual(correct_second_moment_array, second_moment_array)
        self.assertArrayEqual(correct_normalizer_array, normalizer_array)

    def test_compute_weighted_first_moment_array(self):
        """ Weighted first moments, one moment per col of W."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        compute_weighted_first_moment_array(first_moment_array,
                                   self.td.RX,
                                   self.td.mask,
                                   self.td.W,
                                   self.td.N,
                                   self.td.K,
                                   self.td.Q,
                                   self.td.P)
        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTX) / \
                                     np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        self.assertArrayEqual(first_moment_array, correct_first_moment_array)

    def test_compute_weighted_first_and_second_moment_array(self):
        """ Weighted first and second moments, one moment per col of W."""
        first_moment_array = np.zeros((self.td.K, self.td.P))
        second_moment_array = np.zeros((self.td.K, self.td.P))
        compute_weighted_first_and_second_moment_array(first_moment_array,
                                   second_moment_array,
                                   self.td.RX,
                                   self.td.mask,
                                   self.td.W,
                                   self.td.N,
                                   self.td.K,
                                   self.td.Q,
                                   self.td.P)
        correct_first_moment_array = np.dot(self.td.W.T, self.td.RRTX) / \
                                     np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        correct_second_moment_array = np.dot(self.td.W.T, self.td.RRTX**2) / \
                                      np.dot(self.td.W.T, (self.td.RRTX!=0).astype(int))
        self.assertArrayEqual(first_moment_array, correct_first_moment_array)
        self.assertArrayEqual(second_moment_array, correct_second_moment_array)

    def test_apply_mask_to_full_sample(self):
        """ Apply mask[2] to U[1]. """

        compressed_sample = np.zeros(self.td.Q, dtype = np.float64)
        apply_mask_to_full_sample(compressed_sample, 
                                  self.td.U[1],
                                  self.td.mask[2],
                                  self.td.Q)

        correct_compressed_sample = np.array([1,4,5], dtype = np.float64)
        self.assertArrayEqual(compressed_sample, correct_compressed_sample)

if __name__ == '__main__':
    unittest.main()





























