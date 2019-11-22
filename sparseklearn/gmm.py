import numpy as np
from sys import float_info
from .sparsifier import Sparsifier
from .kmeans import KMeans
from scipy.special import logsumexp

from sparseklearn.fastLA import logdet_cov_diag

class GaussianMixture(Sparsifier):

    def fit(self, X = None, HDX = None, RHDX = None, y = None):

        self.fit_sparsifier(X = X, HDX = HDX, RHDX = RHDX)

        #best_lpn = -np.finfo(float).max
        #best_means_ = None

        results = []
        for n in range(self.n_init):
            log_prob_norm, counter = self.fit_single_trial()
            this_run = {'log_prob_norm' : log_prob_norm,
                        'counter' : counter,
                        'means' : self.means_,
                        'covariances' : self.covariances_,
                        'weights' : self.weights_,
                        'converged' : self.converged}
            if self.predict_training_data == True:
                this_run['labels_predicted'] = self.predict(self.RHDX)
            results.append(this_run)

        self.results = results

        # choose the best ones
        best_run_index = np.argmax([d['log_prob_norm'] for d in results])
        self.converged = results[best_run_index]['converged']
        self.means_ = results[best_run_index]['means']
        self.covariances_ = results[best_run_index]['covariances']
        self.weights_ = results[best_run_index]['weights']
        self.log_prob_norm_ = results[best_run_index]['log_prob_norm']
        self.counter = results[best_run_index]['counter']
        if self.predict_training_data:
            self.labels_predicted = results[best_run_index]['labels_predicted']


    def fit_single_trial(self):
        self.converged = False
        self._initialize_parameters()
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

    def predict(self, X):
        """ Predict class for each example in X. Assumes X is preconditioned
        and subsampled if necessary.
        """
        _, logresp, _ = self._estimate_log_prob_resp(self.weights_,
                self.means_, self.covariances_, self.covariance_type )
        return np.argmax(logresp, axis=1)

    def _initialize_means(self):
        """ The ::means_init param will be one of the following three:
                None
                n_components X num_feat_full array of initial means
                n_components X num_feat_full X n_init array of initial means

        This function chooses random means in the first case (according to
        init params), passes the means_init back in the second, and increments
        to the next set of means in the third.

        """
        if self.means_init is None:
            if self.init_params == "kmpp":
                means_init_this_run, _ = self._pick_K_dense_datapoints_kmpp(self.n_components)
            elif self.init_params == "random":
                means_init_this_run, _ = self._pick_K_dense_datapoints_random(self.n_components)
            else:
                raise Exception('init_params must be "kmpp" or "random".')
        elif self.means_init.ndim == 2:
            means_init_this_run = self.means_init
        elif self.means_init.ndim == 3:
            if self.means_init_counter > self.n_init:
                raise Exception('Number of mean inits must equal n_init.')
            means_init_this_run = self.means_init[self.means_init_counter]
            self.means_init_counter += 1
        else:
            raise Exception('means_init must be 2d array, 3d array, or None.')
        return means_init_this_run

    def _initialize_covariances(self, means_init):
        """ Requires that _initialize_means has been run because that function
        sets self.means_init_counter. """
        if self.covariances_init is None:
            resp_bootstrapped = self._init_resp_from_means(means_init)
            _, _, covariances_bootstrapped = \
                self._estimate_gaussian_parameters(resp_bootstrapped, self.covariance_type)
            covariances_init_this_run = covariances_bootstrapped
        # in either of these two cases there is a single covar init
        elif (self.covariances_init.ndim == 1 and self.covariance_type == 'spherical') \
            or (self.covariances_init.ndim == 2 and self.covariance_type == 'diag'):
            covariances_init_this_run = self.covariances_init
        # in either of these two cases there are multiple covar inits.
        elif (self.covariances_init.ndim == 2 and self.covariance_type == 'spherical') \
            or (self.covariances_init.ndim == 3 and self.covariance_type == 'diag'):
            # means_init_counter has been incremented already.
            covariances_init_this_run = self.covariances_init[self.means_init_counter-1]
        else:
            raise Exception('Covariances init of wrong form.')
        return covariances_init_this_run

    def _initialize_weights(self, means_init):
        """ Requires that _initialize_means has been run because that function
        sets self.means_init_counter. """
        if self.weights_init is None:
            resp_bootstrapped = self._init_resp_from_means(means_init)
            weights_bootstrapped, _, _ = \
                self._estimate_gaussian_parameters(resp_bootstrapped, self.covariance_type)
            weights_init_this_run = weights_bootstrapped
        elif self.weights_init.ndim == 1:
            weights_init_this_run = self.weights_init
        elif self.weights_init.ndim == 2:
            # means_init_counter has been incremented already.
            weights_init_this_run = self.weights_init[self.means_init_counter-1]
        else:
            raise Exception('weights init of wrong form.')
        return weights_init_this_run/weights_init_this_run.sum()

    def _initialize_parameters(self):
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
        self.means_ = self._initialize_means()
        self.covariances_ = self._initialize_covariances(self.means_)
        self.weights_ = self._initialize_weights(self.means_)
        self.log_prob_norm_ = -np.finfo(float).max

    def _init_resp(self, init_params, weights_init, means_init, covariances_init):
        if means_init is None:
            if init_params == "kmpp":
                means, _ = self._pick_K_dense_datapoints_kmpp(self.n_components)
            elif init_params == "random":
                means, _ = self._pick_K_dense_datapoints_random(self.n_components)
        elif means_init is not None:
            means = means_init
        if covariances_init is not None:
            print('do this')
        else:
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
        # undo the rescaling due to compression (this is just how the pdf worked out)
        maha_dist_squared *= self.num_feat_comp / self.num_feat_full
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
            means_init = None,
            covariances_init = None,
            weights_init = None,
            n_passes = 1,
            predict_training_data = False,
            **kwargs):

        self.init_params = init_params
        self.n_components = n_components
        self.tol = tol
        self.n_passes = n_passes
        self.n_init = n_init
        self.max_iter = max_iter
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.weights_init = weights_init
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.predict_training_data = predict_training_data
        self.means_init_counter = 0
        self.covariances_init_counter = 0
        self.weights_init_counter = 0
        super(GaussianMixture, self).__init__(**kwargs)
