import numpy as np
from sys import float_info
from .sparsifier import Sparsifier
from .kmeans import KMeans
from scipy.special import logsumexp
from .fastLA import logdet_cov_diag


class GaussianMixture(Sparsifier):

    def fit(self, X = None, HDX = None, RHDX = None, y = None):

        self.fit_sparsifier(X = X, HDX = HDX, RHDX = RHDX)

        best_lpn = -np.finfo(float).max

        best_means_ = None
        for n in range(self.n_init):
            log_prob_norm, counter = self.fit_single_trial()
            if log_prob_norm >= best_lpn:
                best_lpn = log_prob_norm
                best_counter = counter
                best_means_ = self.means_
                best_covariances_ = self.covariances_
                best_weights_ = self.weights_
                best_converged = self.converged

        if best_means_ is not None:
            self.converged = best_converged
            self.means_ = best_means_
            self.covariances_ = best_covariances_
            self.weights_ = best_weights_
            self.log_prob_norm_ = best_lpn
            self.counter = best_counter

        #log_prob_norm, counter = self.fit_single_trial()
        #self.means_, self.covariances_ = self.reconstruct_parameters(self.n_passes)

    def reconstruct_parameters(self, n_passes):
        #TODO: fix covariance inversion
        if n_passes == 1:
            if self.transform in ['dct']:
                means_ = self.invert_HD(self.means_)
                covariances_ = self.covariances_
                #covariances_ = self.invert_HD(self.covariances_)
            else:
                means_ = self.means_
                covariances_ = self.covariances_
        elif n_passes == 2:
            raise ValueError('Cannot currently handle 2-pass')
        else:
            raise ValueError('n_passes needs to be 1 or 2, but is {}'.format(self.n_passes))
        return [means_, covariances_]

    def _predict_training_data(self):
        _, logresp, _ = self._estimate_log_prob_resp(self.weights_, 
                                     self.means_,
                                     self.covariances_,
                                     self.covariance_type)
        return np.argmax(logresp + np.log(self.weights_) ,axis=1)

    def fit_single_trial(self):
        self.converged = False
        self._initialize_parameters(self.init_params, 
                self.means_init, self.covariance_type)
        counter = 0
        log_prob_norm = -np.finfo(float).max
        while not self.converged and counter < self.max_iter:
            # E-step
            log_prob, log_resp, log_prob_norm = self._estimate_log_prob_resp(self.weights_, 
                    self.means_, self.covariances_, self.covariance_type)
            # M-step
            self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
                    np.exp(log_resp), self.covariance_type)
            # convergence check
            self.converged = self._convergence_check(log_prob_norm)
            self.log_prob_norm_ = log_prob_norm
            counter += 1
        return [log_prob_norm, counter]


    def _initialize_parameters(self, init_params, means_init, covariance_type):
        """ Initialize the parameters. Sets self.weights_, self.means_, and 
        self.covariances_. Initializes resp using means_init, or if this is
        None, using the method prescribed by init_params. resp is then used
        to intialize the parameters. Also sets self.log_prob_norm_ to max
        float. 

        Parameters
        ----------

        init_params : {'kmpp', 'random'}

        means_init : nd.array or None

        covariance_type : {'spherical', 'diag'}
        """
        resp = self._init_resp(init_params, means_init) 
        self.weights_, self.means_, self.covariances_ = \
                self._estimate_gaussian_parameters(resp, covariance_type)
        self.log_prob_norm_ = -np.finfo(float).max
        """
        if self.init_params == 'kmeans':
            kmc = KMeans(num_feat_full = self.num_feat_full,
                         num_feat_comp = self.num_feat_comp,
                         num_feat_shared = self.num_feat_shared,
                         num_samp = self.num_samp,
                         transform = self.transform,
                         mask = self.mask,
                         D_indices = self.D_indices,
                         n_clusters = self.n_components, 
                         tol = self.tol,
                         init = self.kmeans_init, 
                         max_iter = self.kmeans_max_iter, 
                         n_passes = self.n_passes, 
                         n_init = 1)
            kmc.fit(X = self.X, HDX = self.HDX, RHDX = self.RHDX)
            resp = np.zeros((self.num_samp, self.n_components))
            resp[np.arange(self.num_samp), kmc.labels_] = 1
            self.means_, self.covariances_ = self._estimate_gaussian_means_and_covariances(resp,
                    self.covariance_type)

        elif self.init_params == 'random':
            resp = rng.rand(self.num_samp, self.n_components)
            resp = resp/resp.sum(axis=1)[:,np.newaxis]
            self.means_, self.covariances_ = self._estimate_gaussian_means_and_covariances(resp,
                    self.covariance_type)
            if self.means_init is not None:
                self.means_ = self.means_init

        else:
            raise ValueError('Unimplemented initialization method: {}'.format(self.init_params))

        weights = self._estimate_gaussian_weights(resp)
        self.weights_ = (weights if self.weights_init is None else self.weights_init)
        """


    def _init_resp(self, init_params, means_init):
        """ Initialize the responsibiltiy matrix. If means_init is
        not None then it is used to compute resp, otherwise init_params
        specifies which method (kmpp or random) to use for sampling means at 
        random. 
        """

        if init_params == "kmpp" and means_init is None:
            means, _ = self._pick_K_dense_datapoints_kmpp(self.n_components)
        elif init_params == "random" and means_init is None:
            means, _ = self._pick_K_dense_datapoints_random(self.n_components)
        elif means_init is not None:
            means = means_init
        resp = self._init_resp_from_means(means)
        return resp


    def _init_resp_from_means(self, means_init):
        """ Initialize the responsibility matrix from dense means by doing hard 
        assignment.

        Parameters
        ----------

        means : nd.array, shape (K,P)
            The dense, transformed array of initial means (from random sampling
            or kmpp).

        Returns
        -------

        resp : nd.array, shape (N,K)
            The responsibility matrix.

        """
        distances = self.pairwise_distances(means_init)
        resp = np.zeros((self.num_samp, self.n_components))
        closest = np.argmin(distances, axis = 1)
        resp[np.arange(self.num_samp), closest] = 1
        return resp


    # parameter and responsibiltiy computation

    # E-step
    def _estimate_log_prob_resp(self, weights, means, covariances, covariance_type):
        log_prob = self._compute_log_prob(means, covariances, covariance_type)
        log_resp, log_prob_norm = self._compute_log_resp(weights, log_prob)
        return [log_prob, log_resp, log_prob_norm]

    def _compute_logdet_array(self, covariances, covariance_type):
        #TODO: consider moving the diag part to sparsifier for consistency
        if covariance_type == 'spherical':
            logdet_vector = self.num_feat_comp * np.log(covariances)
            logdetS = np.tile(logdet_vector, (self.num_samp, 1))
        elif covariance_type == 'diag':
            logdetS = np.zeros((self.num_samp, self.n_components), dtype=np.float64)
            logdet_cov_diag(logdetS,
                            covariances,
                            self.mask,
                            self.num_samp,
                            self.n_components,
                            self.num_feat_comp,
                            self.num_feat_full)
        else:
            raise Exception('covariance_type {} not implemented'.format(covariance_type))
        return logdetS

    def _compute_log_prob(self, means, covariances, covariance_type):
        maha_dist_squared = self.pairwise_mahalanobis_distances(means, 
                                covariances, covariance_type)**2
        logconst = self.num_feat_comp*np.log(2*np.pi)
        logdetS = self._compute_logdet_array(covariances, covariance_type)
        log_prob = -.5 * (logconst + maha_dist_squared + logdetS)
        return log_prob

    def _compute_log_resp(self, weights, log_prob):
        lse = logsumexp(log_prob, b = weights, axis = 1)
        log_resp = np.log(weights) + log_prob - lse[:, np.newaxis]
        log_prob_norm = np.mean(lse)
        return [log_resp, log_prob_norm]
        
    # M-step
    def _estimate_gaussian_parameters(self, resp, covariance_type):
        weights = self._estimate_gaussian_weights(resp)
        means, covariances = self._estimate_gaussian_means_and_covariances(resp, covariance_type)
        return [weights, means, covariances]

    def _estimate_gaussian_means_and_covariances(self, resp, covariance_type):
        if covariance_type == 'diag':
            means, covariances = self.weighted_means_and_variances(resp)
        elif covariance_type == 'spherical':
            means, covariances = self.weighted_means_and_variances(resp)
            covariances = np.mean(covariances, axis=1)
        covariances += self.reg_covar
        return [means, covariances]

    def _estimate_gaussian_weights(self, resp):
        """ 
        Note: sklearn returns the counts instead of weights, i.e., no
        division by self.num_samp. 
        """
        rk = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
        weights = rk/self.num_samp
        return weights
        
    def _convergence_check(self, log_prob_norm):
        with np.errstate(over='raise'):
            try:
                diff = np.abs((log_prob_norm - self.log_prob_norm_)/log_prob_norm)
            except FloatingPointError:
                #overflow from initializing self.log_prob_norm_ to -max float
                diff = np.abs(self.log_prob_norm_)
        if diff < self.tol:
            converged = True
        else:
            converged = False
        return converged

    def __init__(self, n_components = 1, covariance_type = 'spherical', tol = 0.001,
            reg_covar = 1e-06, max_iter = 100, n_init = 1, 
            init_params = 'kmpp', 
            weights_init = None, means_init = None, n_passes = 1,
            **kwargs):

        #kmeans_init = 'random' and 'kmeans_max_iter' = 0
        self.init_params = init_params
        self.n_components = n_components
        self.tol = tol
        self.n_passes = n_passes
        self.n_init = n_init
        self.max_iter = max_iter
        self.means_init = means_init
        self.weights_init = weights_init
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.means_init_counter = 0
        #self.kmeans_max_iter = kmeans_max_iter
        # overwrite kmeans_init for compatibility with the KMeans classifer
        #if means_init is None:
        #    self.kmeans_init = kmeans_init
        #else:
        #    self.kmeans_init = means_init

        super(GaussianMixture, self).__init__(**kwargs)

