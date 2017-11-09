
import numpy as np
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
   
    def fit(self, X):
        self.initialize(X)
        self.initialize_centroids()

    def initialize_centroids(self):
        """ Initialize the centroid guesses. """
        if self.init == 'k-means++':
            centroids = self.initialize_centroids_kmpp()
        if self.init == 'random':
            indices = list(np.random.choice(self.N, self.K, replace = False))
    
    def initialize_centroids_kmpp(self):
        if self.kmpp_resample:
            mask = self.mask[0]
            X_init = self.apply_mask(self.HDX[:], mask, cross_terms = True)

        else:
            mask = self.mask
            X_init = self.HDX_sub

        centroid_indices = np.zeros(self.K, dtype = int)
        centroids = np.zeros((self.K, self.P))

        centroid_indices[0] = np.random.choice(self.N)
        if self.full_init:
            centroids[0] = self.HDX[centroid_indices[0]]
        else:
            centroids[0][self.mask[centroid_indices[0]]] = \
                self.HDX_sub[centroid_indices[0]]

        d_prev = self.pairwise_distances(X_init, centroids[0], 
                mask = mask, transform_Y = "R")[:,0]

        for k in range(1,self.K):
            d_curr = self.pairwise_distances(X_init, Y = centroids[k], mask = mask, 
                     transform_Y = "R")[:,0]
            d_curr = np.min((d_prev, d_curr), axis = 0)
            d_prev = np.copy(d_curr)

            d_curr_sum = d_curr.sum()
            if d_curr_sum > 0:
                centroid_indices[k] = np.random.choice(self.N, p = d_curr/d_curr_sum)
            else:
                available_indices = set(range(self.N)).difference(set(centroid_indices))
                centroid_indices[k] = np.random.choice(list(available_indices))
            if self.full_init:
                centroids[k] = self.HDX[centroid_indices[k]]
            else:
                centroids[k][self.mask[centroid_indices[k]]] = \
                    self.HDX_sub[centroid_indices[k]]
            

        import pdb; pdb.set_trace()

    def __init__(self, n_clusters = 8, init = 'k-means++', full_init = True,
                 kmpp_resample = True, n_init = 10,
                 max_iter = 300, n_passes = 1, **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.K = n_clusters
        self.init = init
        self.full_init = full_init
        self.kmpp_resample = kmpp_resample
        self.n_init = 10
        self.max_iter = 300
        self.n_passes = 1
