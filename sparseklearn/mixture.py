import numpy as np
from .sparsifier import Sparsifier
from .kmeans import KMeans


class GaussianMixture(Sparsifier):

    def fit(self, X, y = None):
        self.fit_sparsifier(X)
        self.initialize_means()

    def initialize_means(self):
        if self.init_params == 'random':
            None
        elif self.init_params == 'kmeans':
            kmc = KMeans(n_clusters = self.n_components, tol = self.tol,
                    init = 'k-means++',
                    full_init = self.full_init, max_iter = self.max_iter,
                    n_passes = self.n_passes, n_init = 1)
            kmc.fit(self) 
            self.means_ = kmc.cluster_centers_

    #def __init__(self, n_clusters = 8, init = 'k-means++', tol = 1e-4, 
    #             full_init = True, n_init = 10, max_iter = 300, 
    #             n_passes = 1, **kwargs):

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
        self.full_init = full_init
        self.max_iter = max_iter

        super(GaussianMixture, self).__init__(**kwargs)
