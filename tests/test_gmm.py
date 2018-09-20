import unittest
import numpy as np
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture import GaussianMixture as GMSKL
from sparseklearn import GaussianMixture

from generate_test_data import DataGenerator

class TestGaussianMixture(unittest.TestCase):

    def assertArrayEqual(self, x, y):
        self.assertTrue(np.allclose(x, y, rtol=1e-6))

    def setUp(self):
        self.td = DataGenerator()
        #gmm = GaussianMixture(n_components = 3, 
        #                num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
        #                num_samp = 4, transform = 'dct', 
        #                D_indices = self.td.D_indices, mask = self.td.mask)
        #self.gmm = gmm

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


    ###########################################################################
    ###########################################################################
    #####                             E-STEP                             ######
    ###########################################################################
    ###########################################################################

    def test__compute_logdet_array(self):
        gmm = GaussianMixture(n_components = 3, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        logdet_spherical = gmm._compute_logdet_array(self.td.spherical_covariances, 'spherical')
        logdet_diag = gmm._compute_logdet_array(self.td.diagonal_covariances, 'diag')
        self.assertArrayEqual(self.td.correct_logdet_spherical, logdet_spherical)
        self.assertArrayEqual(self.td.correct_logdet_diag, logdet_diag)

    def test__compute_log_prob_spherical_no_compression(self):
        """ Compare the log_prob computation to that of sklearn with no 
        compression. Implemented as a precursor to testing it with 
        compression, to follow. Spherical covariances. """
        cov_type = 'spherical'
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type)
        gmm.fit_sparsifier(X = self.td.X)
        means = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        covariances = np.random.rand(gmm.n_components)
        log_prob_test = gmm._compute_log_prob(means, covariances, cov_type)
        precisions = _compute_precision_cholesky(covariances, cov_type)
        log_prob_true = _estimate_log_gaussian_prob(self.td.X, means, precisions, cov_type) 
        self.assertArrayEqual(log_prob_test, log_prob_true)

    def test__compute_log_prob_diagonal_no_compression(self):
        """ Compare the log_prob computation to that of sklearn with no 
        compression. Implemented as a precursor to testing it with 
        compression, to follow. Diagonal covariances. """
        cov_type = 'diag'
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type)
        gmm.fit_sparsifier(X = self.td.X)
        means = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        covariances = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        log_prob_test = gmm._compute_log_prob(means, covariances, cov_type)
        precisions = _compute_precision_cholesky(covariances, cov_type)
        log_prob_true = _estimate_log_gaussian_prob(self.td.X, means, precisions, cov_type) 
        self.assertArrayEqual(log_prob_test, log_prob_true)

    def test__compute_log_prob(self):
        """ This test should probably get implemented eventually. It corresponds 
        to testing the wrapper around fastLA.pairwise_mahalanobis_distances and 
        gmm._compute_logdet_array. Each of these has tests for spherical and
        diagonal cases with sparsification.
        
        Currently we have tests of:

            - component functions in _compute_log_prob
              includes diag and spherical on compressed data with 
              sparsification
            - _compute_log_prob on dense data
              tests against sklearn
        
        I can't currently think of a way to implement a test for this that isn't
        a trivial replication of those earlier tests.
        
        """
        #TODO

    def test__estimate_log_prob_resp_spherical_no_compression(self):
        cov_type = 'spherical'
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type)
        gmm.fit_sparsifier(X = self.td.X)
        means = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        covariances = np.random.rand(gmm.n_components)
        weights = np.random.rand(gmm.n_components)
        weights /= weights.sum()
        log_prob_test, log_resp_test, log_prob_norm_test = gmm._estimate_log_prob_resp(
            weights, means, covariances, cov_type)
        # find skl's values, pretty ugly to do. 
        precisions = _compute_precision_cholesky(covariances, cov_type)
        gmm_skl = GMSKL(n_components = 3, covariance_type = cov_type)
        gmm_skl.means_ = means
        gmm_skl.precisions_cholesky_ = precisions
        gmm_skl.weights_ = weights
        gmm_skl.covariance_type_ = cov_type
        log_prob_norm_true, log_resp_true = gmm_skl._estimate_log_prob_resp(self.td.X)
        # if anything is bad later this overwrite with mean seems suspect:
        log_prob_norm_true = log_prob_norm_true.mean() 
        # now get the log_prob from another function
        log_prob_true = _estimate_log_gaussian_prob(self.td.X, means, precisions, cov_type) 
        # run the tests
        self.assertArrayEqual(log_prob_test, log_prob_true)
        self.assertArrayEqual(log_prob_norm_true, log_prob_norm_test)
        self.assertArrayEqual(log_resp_true, log_resp_test)

    def test__estimate_log_prob_resp_diagonal_no_compression(self):
        cov_type = 'diag'
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type)
        gmm.fit_sparsifier(X = self.td.X)
        means = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        covariances = np.random.rand(gmm.n_components, gmm.num_feat_comp)
        weights = np.random.rand(gmm.n_components)
        weights /= weights.sum()
        log_prob_test, log_resp_test, log_prob_norm_test = gmm._estimate_log_prob_resp(
            weights, means, covariances, cov_type)
        # find skl's values, pretty ugly to do. 
        precisions = _compute_precision_cholesky(covariances, cov_type)
        gmm_skl = GMSKL(n_components = 3, covariance_type = cov_type)
        gmm_skl.means_ = means
        gmm_skl.precisions_cholesky_ = precisions
        gmm_skl.weights_ = weights
        gmm_skl.covariance_type_ = cov_type
        log_prob_norm_true, log_resp_true = gmm_skl._estimate_log_prob_resp(self.td.X)
        # if anything is bad later this overwrite with mean seems suspect:
        log_prob_norm_true = log_prob_norm_true.mean() 
        # now get the log_prob from another function
        log_prob_true = _estimate_log_gaussian_prob(self.td.X, means, precisions, cov_type) 
        # run the tests
        self.assertArrayEqual(log_prob_test, log_prob_true)
        self.assertArrayEqual(log_prob_norm_true, log_prob_norm_test)
        self.assertArrayEqual(log_resp_true, log_resp_test)


    ###########################################################################
    ###########################################################################
    #####                             M-STEP                             ######
    ###########################################################################
    ###########################################################################

    def test__estimate_gaussian_parameters_spherical_no_compression(self):
        """ Test _estiamte_gaussian_parameters against sklearn's
        implementation. Spherical covariances, no compression. 
        """
        cov_type = 'spherical'
        reg_covar = 1e-6
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type, 
                reg_covar = reg_covar)
        gmm.fit_sparsifier(X = self.td.X)
        resp = np.random.rand(gmm.num_samp, gmm.n_components)
        weights_test, means_test, covariances_test = gmm._estimate_gaussian_parameters(resp, cov_type)
        # skl
        counts_true, means_true, covariances_true = _estimate_gaussian_parameters(
                self.td.X, resp, reg_covar, cov_type) 
        # skl returns counts instead of weights. 
        weights_true = counts_true / gmm.num_samp

        self.assertArrayEqual(weights_test, weights_true)
        self.assertArrayEqual(means_test, means_true)
        self.assertArrayEqual(covariances_test, covariances_true)

    def test__estimate_gaussian_parameters_diagonal_no_compression(self):
        """ Test _estiamte_gaussian_parameters against sklearn's
        implementation. Diagonal covariances, no compression. 
        """
        cov_type = 'diag'
        reg_covar = 1e-6
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5, 
                num_feat_comp = 5, num_feat_shared = 5, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type, 
                reg_covar = reg_covar)
        gmm.fit_sparsifier(X = self.td.X)
        resp = np.random.rand(gmm.num_samp, gmm.n_components)
        weights_test, means_test, covariances_test = gmm._estimate_gaussian_parameters(resp, cov_type)
        # skl
        counts_true, means_true, covariances_true = _estimate_gaussian_parameters(
                self.td.X, resp, reg_covar, cov_type) 
        # skl returns counts instead of weights. 
        weights_true = counts_true / gmm.num_samp

        self.assertArrayEqual(weights_test, weights_true)
        self.assertArrayEqual(means_test, means_true)
        self.assertArrayEqual(covariances_test, covariances_true)

    def test__estimate_gaussian_means_and_covariances_diagonal_no_compression(self):
        """ Test _estimate_gaussian_means_and_covariances against hard-coded 
        example. Should be redundant with test__estimate_gaussian_parameters_*
        tests above, which test against sklearn's results. """

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
        """ Weights are testsed in test__estimate_gaussian_parameters_* above.
        Should not need to implement this unless we want to further test on
        compressed case. 
        """
        #TODO
        return 1


    ###########################################################################
    ###########################################################################
    #####                         Initialization                         ######
    ###########################################################################
    ###########################################################################


    def test__init_resp_from_means(self):
        gmm = GaussianMixture(n_components = 3, 
                        num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
                        num_samp = 4, transform = 'dct', 
                        D_indices = self.td.D_indices, mask = self.td.mask)
        gmm.fit_sparsifier(X=self.td.X)
        resp_test = gmm._init_resp_from_means(self.td.U)
        resp_correct = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]], dtype = int)
        self.assertArrayEqual(resp_test, resp_correct)

    def test__init_resp_kmeans(self):
        """ Does not compare against true result, instead checks that
        responsibility matrix is of correct form and has rows of all
        zeros with a single one.
        """
        init_params = 'kmeans'
        gmm = GaussianMixture(n_components = 3, 
            num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
            num_samp = 4, transform = 'dct', 
            D_indices = self.td.D_indices, mask = self.td.mask,
            init_params = init_params)
        gmm.fit_sparsifier(HDX=self.td.HDX)
        resp = gmm._init_resp()
        # check shape
        self.assertArrayEqual(resp.shape, [gmm.num_samp, gmm.n_components])
        # check number of nonzeros
        nonzeros_per_row_test = (np.abs(resp)>0).sum(axis=1)
        nonzeros_per_row_correct = np.ones(gmm.num_samp)
        self.assertArrayEqual(nonzeros_per_row_test, 
                              nonzeros_per_row_correct)
        # check row sums
        rowsum_test = resp.sum(axis=1)
        rowsum_correct = np.ones(gmm.num_samp)
        self.assertArrayEqual(rowsum_test, rowsum_correct)

    def test__init_resp_random(self):
        """ Does not compare against true result, instead checks that
        responsibility matrix is of correct form and has rows of all
        zeros with a single non-zero.
        """
        init_params = 'random'
        gmm = GaussianMixture(n_components = 3, 
            num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
            num_samp = 4, transform = 'dct', 
            D_indices = self.td.D_indices, mask = self.td.mask,
            init_params = init_params)
        gmm.fit_sparsifier(HDX=self.td.HDX)
        resp = gmm._init_resp()
        # check shape
        self.assertArrayEqual(resp.shape, [gmm.num_samp, gmm.n_components])
        # check number of nonzeros
        nonzeros_per_row_test = (np.abs(resp)>0).sum(axis=1)
        nonzeros_per_row_correct = np.ones(gmm.num_samp)
        self.assertArrayEqual(nonzeros_per_row_test, 
                              nonzeros_per_row_correct)
        # check row sums
        rowsum_test = resp.sum(axis=1)
        rowsum_correct = np.ones(gmm.num_samp)
        self.assertArrayEqual(rowsum_test, rowsum_correct)

    ###########################################################################
    ###########################################################################
    #####                         To be Assigned                         ######
    ###########################################################################
    ###########################################################################



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
        gmm.fit(X=X)
        correct_means = np.array([[0, 0, 1, 0],
                                  [1, 0, 0, 0]], dtype = np.float64)
        correct_covariances = np.ones_like(correct_means)*reg_covar
        self.assertArrayEqual(gmm.means_, correct_means)
        self.assertArrayEqual(gmm.covariances_, correct_covariances)

    def test__predict_training_data(self):
        #TODO
        X = np.array([[0,1,0,0],
                      [1,0,0,0],
                      [0,0,1,2]], dtype = np.float64)
        gmm = GaussianMixture(n_components = 2, covariance_type = 'diag',
                num_feat_full = 4, num_feat_comp = 2,
                num_feat_shared = 1, num_samp = 3, transform = None, D_indices = None,
                mask = None, reg_covar = 1e-6, init_params = 'random', max_iter = 5)
        gmm.fit(X=X)
        labels = gmm._predict_training_data()

if __name__ == '__main__':
    unittest.main()

