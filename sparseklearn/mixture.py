import numpy as np
from sys import float_info
from .sparsifier import Sparsifier
from .kmeans import KMeans
from scipy.special import logsumexp


class GaussianMixture(Sparsifier):

    def fit(self, X, y = None):

        self.fit_sparsifier(X)
        # this method requires the mask inverse
        self.mask_inverse = self.invert_mask_bool()

        best_lpn = -float_info.max

        best_means_ = None
        for n in range(self.n_init):
            log_prob_norm, counter = self.fit_single_trial()
            print([log_prob_norm, counter])
            if log_prob_norm >= best_lpn:
                print('Got a better fit')
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
        self.means_, self.covariances_ = self.reconstruct_parameters(self.n_passes)

    def reconstruct_parameters(self, n_passes):
        if n_passes == 1:
            if self.use_ROS:
                means_ = self.invert_ROS(self.means_, self.D_indices)
                covariances_ = self.invert_ROS(self.covariances_, self.D_indices)
            else:
                means_ = self.means_
                covariances_ = self.covariances_
        elif n_passes == 2:
            raise ValueError('Cannot currently handle 2-pass')
        else:
            raise ValueError('n_passes needs to be 1 or 2, but its {}'.format(self.n_passes))
        return [means_, covariances_]

    def fit_single_trial(self):
        self.converged = False
        self._initialize_parameters()
        counter = 0
        log_prob_norm = -float_info.max
        while not self.converged and counter < self.max_iter:
            # E-step
            log_prob, log_resp, log_prob_norm = self._estimate_log_prob_resp(self.weights_, 
                    self.means_, self.covariances_)
            # M-step
            self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
                    np.exp(log_resp))
            # convergence check
            self.converged = self._convergence_check(log_prob_norm)
            self.log_prob_norm_ = log_prob_norm
            counter += 1
        return [log_prob_norm, counter]


    def _initialize_parameters(self):

        # compute the means and responsibilities first
        if self.init_params == 'kmeans':
            kmc = KMeans(n_clusters = self.n_components, tol = self.tol,
                    init = self.kmeans_init, full_init = self.full_init, 
                    max_iter = self.kmeans_max_iter, 
                    use_ROS = self.use_ROS,
                    n_passes = self.n_passes, n_init = 1)
            kmc.fit(self)
            resp = np.zeros((self.N, self.n_components))
            resp[np.arange(self.N), kmc.labels_] = 1
            rk, rkd = self._estimate_rkd(resp)
            #self.means_ = kmc.cluster_centers_
            self.means_ = self._estimate_gaussian_means(resp, rk, rkd)
            self.kmc = kmc

        elif self.init_params == 'random':
            resp = np.random.rand(self.N, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
            rk, rkd = self._estimate_rkd(resp)
            # only overwrite the computed means if init is random
            # (note: this may lead to unexpected behavior in kmeans init, 
            # should take a look at that)
            if self.means_init is None:
                self.means_ = self._estimate_gaussian_means(resp, rk, rkd)
            else:
                self.means_ = (self.apply_ROS(self.means_init) if self.use_ROS else self.means_init)

        else:
            raise ValueError('Unimplemented initialization method: {}'.format(self.init_params))

        # compute and assign the covariances, overwriting if initialized
        if self.precisions_init is None:
            self.covariances_ = self._estimate_gaussian_covariances(resp, rkd, 
                    self.means_)
        else:
            self.covariances_ = (self.apply_ROS(1/(self.precisions_init + self.reg_covar)) 
                    if self.use_ROS else 1/self.precisions_init)

        # compute and assign the weights, overwriting if initialized
        weights = self._estimate_gaussian_weights(rk)
        self.weights_ = (weights if self.weights_init is None else self.weights_init)


        self.log_prob_norm_ = -float_info.max


    # parameter and responsibiltiy computation

    # E-step
    def _estimate_log_prob_resp(self, weights, means, cov):
        # compute the log probabilities
        const = self.M * np.log(2*np.pi)
        #det = np.sum(np.log(cov), axis=1)
        detn = self.mask_inverse.T.dot(np.log(cov).T)
        S = self.pairwise_distances(Y = means, W = 1/cov)**2
        log_prob = -.5 * (const + detn + S)
        # compute the log responsibilities
        lse = logsumexp(log_prob, b = weights, axis = 1)
        log_resp = np.log(weights) + log_prob - lse[:, np.newaxis]
        log_prob_norm = np.mean(lse)
        print(log_prob_norm)
        return [log_prob, log_resp, log_prob_norm]
        
    # M-step
    def _estimate_gaussian_parameters(self, resp):
        rk, rkd = self._estimate_rkd(resp)
        weights = self._estimate_gaussian_weights(rk)
        means = self._estimate_gaussian_means(resp, rk, rkd)
        covariances = self._estimate_gaussian_covariances(resp, rkd, means)
        return [weights, means, covariances]

    # subroutines for the M-step
    def _estimate_rkd(self, resp):
        rk = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
        rkd = self.mask_inverse.dot(resp).transpose() + \
              10 * np.finfo(resp.dtype).eps
        return [rk, rkd]

    def _estimate_gaussian_means(self, resp, rk, rkd):
        means = self.polynomial_combination(resp)
        means /= rkd
        return means

    def _estimate_gaussian_weights(self, rk):
        weights = rk/self.N
        return weights

    def _estimate_gaussian_covariances(self, resp, rkd, means):
        covariances = self.polynomial_combination(resp, power = 2)
        covariances /= rkd
        covariances += - means**2 + self.reg_covar
        if np.any(covariances <= 0):
            raise Exception('Something is wrong; got a negative variance.')
        return covariances
        
    def _convergence_check(self, log_prob_norm):
        diff = np.abs((log_prob_norm - self.log_prob_norm_)/log_prob_norm)
        if diff < self.tol:
            converged = True
        else:
            converged = False
        return converged

    def __init__(self, n_components = 1, covariance_type = 'full', tol = 0.001,
            reg_covar = 1e-06, max_iter = 100, n_init = 1, 
            init_params = 'kmeans', kmeans_init = 'random', kmeans_max_iter = 0, 
            weights_init = None, means_init = None, 
            precisions_init = None, random_state = None, warm_start = False,
            full_init = True, n_passes = 1,
            **kwargs):

        self.init_params = init_params
        self.n_components = n_components
        self.tol = tol
        self.n_passes = n_passes
        self.n_init = n_init
        self.full_init = full_init
        self.max_iter = max_iter
        self.means_init = means_init
        self.weights_init = weights_init
        self.precisions_init = precisions_init
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.kmeans_max_iter = kmeans_max_iter
        # overwrite kmeans_init for compatibility with the KMeans classifer
        if means_init is None:
            self.kmeans_init = kmeans_init
        else:
            self.kmeans_init = means_init

        super(GaussianMixture, self).__init__(**kwargs)
