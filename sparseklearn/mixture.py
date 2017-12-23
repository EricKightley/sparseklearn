import numpy as np
from sys import float_info
from .sparsifier import Sparsifier
from .kmeans import KMeans
from scipy.special import logsumexp


class GaussianMixture(Sparsifier):

    def fit(self, X, y = None):

        self.fit_sparsifier(X)

        best_lpn = -float_info.max
        self.converged = False
        
        for n in range(self.n_init):
            log_prob_norm, converged, counter = self.fit_single_trial()
            import pdb; pdb.set_trace()
            if log_prob_norm > best_lpn:
                best_lpn = log_prob_norm
                final_counter = counter
                means_ = self.means_
                covariances_ = self.covariances_
                weights_ = self.weights_

        self.means_ = means_
        self.covariances_ = covariances_
        self.weights_ = weights_
        self.log_prob_norm = best_lpn
        self.counter = final_counter

        #self.means_, self.covariances_ = self.reconstruct_parameters(self.n_passes)
        self.means_, _ = self.reconstruct_parameters(self.n_passes)

    def reconstruct_parameters(self, n_passes):
        if n_passes == 1 and self.use_ROS:
            means_ = self.invert_ROS(self.means_, self.D_indices)
            covariances_ = self.invert_ROS(self.covariances_, self.D_indices)
        return [means_, covariances_]

    def fit_single_trial(self):
        self.initialize_parameters()
        counter = 0
        while not self.converged and counter < self.max_iter:
            counter += 1
            log_prob_norm, log_resp = self._e_step()
            self._convergence_check(log_prob_norm)
            self._m_step(log_resp)
            self.log_prob_norm = log_prob_norm
        self.log_prob_norm, self.log_resp = self._e_step()
        return [self.log_prob_norm, self.converged, counter]


    def initialize_parameters(self):
        # initialize the means
        # if we didn't pass it the initial means, compute them...
        if self.means_init is None:
            # ... either randomly ...
            if self.init_params == 'random':
                self.means_, _ = self.pick_K_datapoints(self.n_components)
            # ... or using kmeans
            elif self.init_params == 'kmeans':
                kmc = KMeans(n_clusters = self.n_components, tol = self.tol,
                        init = 'k-means++',
                        full_init = self.full_init, max_iter = self.max_iter,
                        n_passes = 2, n_init = 10)
                kmc.fit(self) 
                print('fitting kmeans')
                if self.use_ROS:
                    self.means_ = kmc.apply_ROS(kmc.cluster_centers_)
                else:
                    self.means_ = kmc.cluster_centers_
                self.kmc = kmc
        # if we did pass it initial means then use these
        else:
            self.means_ = means_init

        # initialize the weights
        # if we didn't pass it the initial weights, compute them...
        if self.weights_init is None:
            # ... either uniformly ...
            if self.init_params == 'random':
                self.weights_ = np.ones(self.n_components)/self.n_components
            # ... or using kmeans
            elif self.init_params == 'kmeans':
                self.weights_ = kmc.n_per_cluster
                self.weights_ = self.weights_ / np.sum(self.weights_)
        # if we did pass it initial weights then use these
        else:
            self.weights_ = weights_init

        # initialize the covariance matrix

        if self.covariance_type != 'diag':
            raise Exception('Currently can only handle diagonal covariances')
        if self.covariance_type == 'diag':
            if self.init_params == 'kmeans':
                #TODO : make this depend on kmeans
                self.covariances_ = np.ones((self.n_components, self.P))
            if self.init_params == 'random':
                #TODO : is there a better initialization?
                self.covariances_ = np.ones((self.n_components, self.P))
                #TODO : this uses the dense data, for testing only. This must be fixed!
                #datacov = np.diag(np.cov(self.HDX))
                #self.covariances_ = np.array([datacov for k in
                #        range(self.n_components)])
        self.log_prob_norm = -float_info.max
       

    # parameter and responsibiltiy computation

    def _estimate_log_gaussian_prob(self,means,covariances):
        if self.covariance_type == 'diag':
            logdetSig = np.log(covariances).sum(axis=1)
            logprob = -.5*(self.P*np.log(2*np.pi) + \
                      logdetSig + \
                      self.pairwise_distances(Y=means, W = 1/covariances)**2)
        else:
            raise Exception('Currently can only handle diagonal covariances')
        return logprob

    def _estimate_weighted_log_prob(self):
        return self._estimate_log_gaussian_prob(self.means_, self.covariances_) + \
               np.log(self.weights_)

    def _estimate_log_prob_resp(self):
        weighted_log_prob = self._estimate_weighted_log_prob()
        log_prob_norm = logsumexp(weighted_log_prob, axis = 1)
        with np.errstate(under='ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return [log_prob_norm, log_resp]

    def _e_step(self):
        log_prob_norm, log_resp = self._estimate_log_prob_resp()
        return [np.mean(log_prob_norm), log_resp]


    def _estimate_gaussian_parameters(self, resp):
        nk = np.sum(resp,axis=0) + 10 * np.finfo(resp.dtype).eps
        means = self._estimate_gaussian_means(resp,nk)
        covariances = self._estimate_gaussian_covariances_diag(resp, nk, means)
        return [nk, means, covariances]

    def _estimate_gaussian_means(self, resp, nk):
        means = np.zeros((self.n_components,self.P))
        for k in range(self.n_components):
            for n in range(self.N):
                means[k][self.mask[n]] += self.HDX_sub[n]*resp[n,k]
        means *= 1/nk[:,np.newaxis]
        return means

    def _estimate_gaussian_covariances_diag(self, resp, nk, means):
        covariances = np.zeros((self.n_components, self.P))
        for n in range(self.N):
            for k in range(self.n_components):
                    covariances[k][self.mask[n]] += (self.HDX_sub[n] - means[k][self.mask[n]])**2
        covariances *= 1/nk[:,np.newaxis] + self.reg_covar
        return covariances


    def _m_step(self, logresp):
        # needs resp set
        self.weights_, self.means_, self.covariances_ = \
                self._estimate_gaussian_parameters(np.exp(logresp))
        self.weights_ /= self.N

    def _convergence_check(self, log_prob_norm):
        if log_prob_norm > 1e-6:
            diff = np.abs((log_prob_norm - self.log_prob_norm)/log_prob_norm)
            if diff < self.tol:
                self.converged = True


    def __init__(self, n_components = 1, covariance_type = 'full', tol = 0.001,
            reg_covar = 1e-06, max_iter = 100, n_init = 1, 
            init_params = 'kmeans', weights_init = None, means_init = None, 
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
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar

        super(GaussianMixture, self).__init__(**kwargs)
