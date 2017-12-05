import numpy as np
#from sys import float_info
from .sparsifier import Sparsifier
from .kmeans import KMeans


class GaussianMixture(Sparsifier):

    def fit(self, X, y = None):

        self.fit_sparsifier(X)
        best_llh = 0
        
        for n in range(self.n_init):
            llh, c = self.fit_single_trial()
            if llh > best_llh:
                best_llh = llh
                means_ = self.means_
                covariances_ = self.covariances_
                weights_ = self.weights_

        self.means_ = means_
        self.covariances_ = covariances_
        self.weights_ = weights_
        self.loglikelihood = best_llh


    def fit_single_trial(self):
        self.initialize_parameters()
        counter = 0
        while not self.converged and counter < self.max_iter:
            self._e_step()
            self._m_step()
            self._convergence_check()
        return [self.loglikelihood, self.converged]


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
                        n_passes = self.n_passes, n_init = 1)
                kmc.fit(self) 
                self.means_ = kmc.cluster_centers_
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

        self.converged = False
        self.loglikelihood = 0.0

    # parameter and responsibiltiy computation

    def _estimate_gaussian_prob(self, means, covariances):
        if self.covariance_type == 'diag':
            import pdb; pdb.set_trace()
            det_sigma = np.product(covariances, axis = 1)
            d = self.pairwise_distances(Y=means, W = 1/covariances)**2
            prob = 1/np.sqrt(2*np.pi*det_sigma) * np.exp(-.5 * d)
        else:
            raise Exception('Currently can only handle diagonal covariances')
        return prob

    def _estimate_resp(self,prob,weights):
        resp = np.zeros((self.N, self.n_components))
        denoms = np.dot(prob,weights)
        resp = (prob.T / denoms).T
        resp *= weights
        return resp

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
        dsquared= self.pairwise_distances(Y = means)**2
        return np.sum(resp*dsquared, axis=0) / nk + self.reg_covar


    # core methods
    def _e_step(self):
        # needs means, covariances, weights set
        self.prob_ = self._estimate_gaussian_prob(self.means_,self.covariances_)
        self.resp_ = self._estimate_resp(self.prob_, self.weights_)

    def _m_step(self):
        # needs resp set
        self.weights_, self.means_, self.covariances_ = \
                self._estimate_gaussian_parameters(self.resp_)
        self.weights_ /= self.N

    def _convergence_check(self):
        loglikelihood = np.log(np.dot(self.prob_, self.weights_)).sum()
        if np.abs((self.loglikelihood - loglikelihood)/loglikelihood) < self.tol:
            self.converged = True
        self.loglikelihood = loglikelihood


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
