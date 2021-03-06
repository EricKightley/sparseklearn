import unittest
import numpy as np
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture import GaussianMixture as GMSKL
from sparseklearn import GaussianMixture

from tests import DataGenerator

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
        self.assertArrayEqual(self.td.mask, gmm.mask)
        self.assertEqual(self.td.N, gmm.num_samp)
        self.assertEqual(self.td.Q, gmm.num_feat_comp)
        self.assertEqual(self.td.P, gmm.num_feat_full)

    def instantiate_standard_gmm(self, random_state):
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        random_state = random_state)
        return gmm

    ###########################################################################
    ###########################################################################
    #####                             E-STEP                             ######
    ###########################################################################
    ###########################################################################

    def test_pairwise_mahalanobis_distances(self):
        """ pairwise_mahalanobis_distances is a Sparsifier function, but a use
        case here made me suspect that it's wrong. To confirm this I'm putting
        a test here first. Will need to make a new one (and probably ammend
        existing ones that are currently passing but shouldn't be) in that
        test suite once I am convinced it's the culprit. """

        cov_type = 'spherical'
        rs = np.random.RandomState(10)
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5,
                num_feat_comp = 3, num_feat_shared = 2, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type,
                random_state = rs)
        gmm.fit_sparsifier(X = self.td.X)
        means = rs.rand(gmm.n_components, gmm.num_feat_full)
        covariances = rs.rand(gmm.n_components)
        mahadist_test = gmm.pairwise_mahalanobis_distances(means, covariances,
            cov_type)**2
        #undo the rescaling due to compression
        mahadist_test *= gmm.num_feat_comp/gmm.num_feat_full

        mahadist_true = np.zeros_like(mahadist_test)
        for data_ind in range(gmm.num_samp):
            for comp_ind in range(gmm.n_components):
                mahadist_true[data_ind, comp_ind] = 1/covariances[comp_ind] * \
                    np.linalg.norm(gmm.RHDX[data_ind] -
                        means[comp_ind][gmm.mask[data_ind]])**2

        self.assertArrayEqual(mahadist_test, mahadist_true)

    def test_hand_computation_of_log_prob_vs_sklearn(self):
        """ Something seems wrong with my mahadist computation. Before digging
        further into the C library to find the error, I want to make sure that
        the results I think it should give are right. One way to gather
        evidence in favor of this conclusion is to use the result in the
        computation of the log probability (this is what led me here in the
        first place). This test does so, and consequently doesn't actually
        test any of the code in gmm.py. For this to work the mask must be
        entirely shared. """
        cov_type = 'spherical'
        rs = np.random.RandomState(10)
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5,
                num_feat_comp = 3, num_feat_shared = 3, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type,
                random_state = rs)
        gmm.fit_sparsifier(X = self.td.X)
        means = rs.rand(gmm.n_components, gmm.num_feat_full)
        covariances = rs.rand(gmm.n_components)
        precisions = _compute_precision_cholesky(covariances, cov_type)

        # this is where we need the mask to be shared, so that all mask rows
        # equal mask[0]
        masked_means = means[:, gmm.mask[0]]
        log_prob_true = _estimate_log_gaussian_prob(gmm.RHDX, masked_means,
                precisions, cov_type)

        log_prob_test = np.zeros((gmm.num_samp, gmm.n_components))
        for data_ind in range(gmm.num_samp):
            for comp_ind in range(gmm.n_components):
                test_const = gmm.num_feat_comp * np.log(2*np.pi)
                test_logdet = gmm.num_feat_comp * np.log(covariances[comp_ind])
                test_mahadist = 1/covariances[comp_ind] * \
                    np.linalg.norm(gmm.RHDX[data_ind] -
                        means[comp_ind][gmm.mask[data_ind]])**2
                log_prob_test[data_ind, comp_ind] = -.5*(test_const + \
                    test_logdet + test_mahadist)
        self.assertArrayEqual(log_prob_test, log_prob_true)


    def test__compute_logdet_array_spherical(self):
        """ Test spherical logdet under compression on an example
        computed here. Redundant with test__compute_logdet_array below but was
        implemented to confirm that test is correct. """
        cov_type = 'spherical'
        rs = np.random.RandomState(10)
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5,
                num_feat_comp = 3, num_feat_shared = 2, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type,
                random_state = rs)
        gmm.fit_sparsifier(X = self.td.X)
        means = rs.rand(gmm.n_components, gmm.num_feat_full)
        covariances = rs.rand(gmm.n_components)

        logdet_test = gmm._compute_logdet_array(covariances, 'spherical')
        logdet_true = gmm.num_feat_comp * np.log(covariances)
        logdet_true = np.tile(logdet_true, (gmm.num_samp, 1))
        self.assertArrayEqual(logdet_test, logdet_true)

    def test__compute_logdet_array(self):
        """ Test spherical and diagonal on hard-coded results. """
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

    def test__compute_log_prob_spherical_shared_compression(self):
        """ Compare the log_prob computation to that of sklearn with
        shared compression. Spherical covariances. """
        cov_type = 'spherical'
        rs = np.random.RandomState(10)
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5,
                num_feat_comp = 3, num_feat_shared = 3, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type,
                random_state = rs)
        gmm.fit_sparsifier(X = self.td.X)
        means = rs.rand(gmm.n_components, gmm.num_feat_full)
        covariances = rs.rand(gmm.n_components)
        log_prob_test = gmm._compute_log_prob(means, covariances, cov_type)

        log_prob_true = np.zeros((gmm.num_samp, gmm.n_components))
        for data_ind in range(gmm.num_samp):
            for comp_ind in range(gmm.n_components):
                true_const = gmm.num_feat_comp * np.log(2*np.pi)
                true_logdet = gmm.num_feat_comp * np.log(covariances[comp_ind])
                true_mahadist = 1/covariances[comp_ind] * \
                    np.linalg.norm(gmm.RHDX[data_ind] -
                        means[comp_ind][gmm.mask[data_ind]])**2
                log_prob_true[data_ind, comp_ind] = -.5*(true_const + \
                    true_logdet + true_mahadist)
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

    # test failing
    def test__estimate_log_prob_resp_spherical_shared_compression(self):
        rs = np.random.RandomState(11)
        cov_type = 'spherical'
        gmm = GaussianMixture(n_components = 3, num_feat_full = 5,
                num_feat_comp = 3, num_feat_shared = 3, num_samp = 4, transform = None,
                mask = None, D_indices = None, covariance_type = cov_type,
                random_state = rs)
        gmm.fit_sparsifier(X = self.td.X)
        means = rs.rand(gmm.n_components, gmm.num_feat_full)
        covariances = rs.rand(gmm.n_components)
        weights = rs.rand(gmm.n_components)
        weights /= weights.sum()
        log_prob_test, log_resp_test, log_prob_norm_test = gmm._estimate_log_prob_resp(
            weights, means, covariances, cov_type)
        # find skl's values, pretty ugly to do.
        precisions = _compute_precision_cholesky(covariances, cov_type)
        gmm_skl = GMSKL(n_components = 3, covariance_type = cov_type)
        # we need the mask to be shared so that we can use mask[0] on all means
        gmm_skl.means_ = means[:, gmm.mask[0]]
        gmm_skl.precisions_cholesky_ = precisions
        gmm_skl.weights_ = weights
        gmm_skl.covariance_type_ = cov_type
        log_prob_norm_true, log_resp_true = gmm_skl._estimate_log_prob_resp(gmm.RHDX)
        # if anything is bad later this overwrite with mean seems suspect:
        log_prob_norm_true = log_prob_norm_true.mean()
        # now get the log_prob from another function
        log_prob_true = _estimate_log_gaussian_prob(gmm.RHDX, gmm_skl.means_,
                        precisions, cov_type)
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

    def test__initialize_means_case1(self):
        """ means_init is a 2D array.
        """
        random_state = np.random.RandomState(12)
        means_init_true = random_state.rand(self.td.K, self.td.P)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init_test = gmm._initialize_means()
        self.assertArrayEqual(means_init_test, means_init_true)

    def test__initialize_means_case2(self):
        """ means_init is a 3D array.
        """
        random_state = np.random.RandomState(12)
        n_init = 3
        means_init_true = random_state.rand(n_init, self.td.K, self.td.P)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        n_init = n_init,
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        # first one is discarded for this test
        _ = gmm._initialize_means()
        # this should recover the second one
        means_init_test = gmm._initialize_means()
        self.assertArrayEqual(means_init_test, means_init_true[1])

    def test__initialize_means_case3(self):
        """ means_init is None, init_params is 'kmpp'.
        Only checks that the initialized means are of the correct shape.
        """
        random_state = np.random.RandomState(12)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = None,
                        init_params = 'kmpp',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init_shape_test = gmm._initialize_means().shape
        means_init_shape_true = np.array([self.td.K, self.td.P])
        self.assertArrayEqual(means_init_shape_test, means_init_shape_true)

    def test__initialize_means_case4(self):
        """ means_init is None, init_params is 'random'.
        Only checks that the initialized means are of the correct shape.
        """
        random_state = np.random.RandomState(12)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = None,
                        init_params = 'random',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init_shape_test = gmm._initialize_means().shape
        means_init_shape_true = np.array([self.td.K, self.td.P])
        self.assertArrayEqual(means_init_shape_test, means_init_shape_true)

    def test__initialize_covariances_case1(self):
        """ spherical covariance, 1 init.
        """
        random_state = np.random.RandomState(12)
        means_init_true = random_state.rand(self.td.K, self.td.P)
        covariances_init_true = random_state.rand(self.td.K)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        covariances_init = covariances_init_true,
                        covariance_type = 'spherical',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init = gmm._initialize_means()
        covariances_init_test = gmm._initialize_covariances(means_init)
        self.assertArrayEqual(covariances_init_test, covariances_init_true)

    def test__initialize_covariances_case2(self):
        """ diagonal covariance, 1 init.
        """
        random_state = np.random.RandomState(12)
        means_init_true = random_state.rand(self.td.K, self.td.P)
        covariances_init_true = random_state.rand(self.td.K, self.td.P)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        covariances_init = covariances_init_true,
                        covariance_type = 'diag',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init = gmm._initialize_means()
        covariances_init_test = gmm._initialize_covariances(means_init)
        self.assertArrayEqual(covariances_init_test, covariances_init_true)

    def test__initialize_covariances_case3(self):
        """ No covariances given, just check shape.
        """
        random_state = np.random.RandomState(12)
        means_init_true = random_state.rand(self.td.K, self.td.P)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        covariances_init = None,
                        covariance_type = 'diag',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        means_init = gmm._initialize_means()
        covariances_init_test = gmm._initialize_covariances(means_init)
        true_shape = np.array((self.td.K, self.td.P))
        self.assertArrayEqual(covariances_init_test.shape, true_shape)

    def test__initialize_covariances_case4(self):
        """ diagonal covariance, multi-init.
        """
        random_state = np.random.RandomState(12)
        n_init = 3
        means_init_true = random_state.rand(n_init, self.td.K, self.td.P)
        covariances_init_true = random_state.rand(n_init, self.td.K, self.td.P)
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        covariances_init = covariances_init_true,
                        covariance_type = 'diag',
                        n_init = n_init,
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        # init means twice to cycle covariances
        _ = gmm._initialize_means()
        means_init = gmm._initialize_means()
        covariances_init_test = gmm._initialize_covariances(means_init)
        self.assertArrayEqual(covariances_init_test, covariances_init_true[1])

    def test__initialize_weights_case3(self):
        """ multi-init
        """
        random_state = np.random.RandomState(12)
        n_init = 3
        means_init_true = random_state.rand(n_init, self.td.K, self.td.P)
        weights_init_true = random_state.rand(n_init, self.td.K)
        weights_init_true /= weights_init_true.sum(axis=1)[:,np.newaxis]
        gmm = GaussianMixture(n_components = self.td.K,
                        num_feat_full = self.td.P,
                        num_feat_comp = self.td.Q,
                        num_feat_shared = self.td.Qs,
                        num_samp = self.td.N,
                        transform = self.td.transform,
                        D_indices = self.td.D_indices,
                        mask = self.td.mask,
                        means_init = means_init_true,
                        weights_init = weights_init_true,
                        n_init = n_init,
                        covariance_type = 'diag',
                        random_state = random_state)
        gmm.fit_sparsifier(X=self.td.X)
        # init means twice to cycle covariances
        _ = gmm._initialize_means()
        means_init = gmm._initialize_means()
        weights_init_test = gmm._initialize_weights(means_init)
        self.assertArrayEqual(weights_init_test, weights_init_true[1])

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

    def test__initialize_parameters(self):
        """ Only tests if it runs. """
        init_params = 'random'
        means_init = None
        gmm = GaussianMixture(n_components = 3,
            num_feat_full = 5, num_feat_comp = 3, num_feat_shared = 1,
            num_samp = 4, transform = 'dct',
            D_indices = self.td.D_indices, mask = self.td.mask,
            init_params = init_params,
            means_init = means_init)
        gmm.fit_sparsifier(HDX=self.td.HDX)
        gmm._initialize_parameters()

    ###########################################################################
    ###########################################################################
    #####                              Fit                               ######
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

if __name__ == '__main__':
    unittest.main()
