
import numpy as np
from sys import float_info
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
   
    def fit(self, X):
        self.initialize(X)
        self.initialize_centroids()

    def initialize_centroids(self):
        """ Initialize the centroid guesses. """
        if self.init == 'k-means++':
            centroids, centroid_indices = self.initialize_centroids_kmpp()
        elif self.init == 'random':
            centoids_, centroid_indices = self.initialize_centroids_random()
        elif type(self.init) is np.ndarray:
            centroids = self.init
            centroid_indices = []
        import pdb; pdb.set_trace()
        return [centroids, centroid_indices]
   
    def initialize_centroids_random(self):
        # pick K data points at random uniformly
        centroid_indices = np.random.choice(self.N, self.K, replace = False)
        centroid_indices.sort()
        # assign the centroids as dense members of HDX ...
        if self.full_init:
            centroids = np.array(self.HDX[centroid_indices])
        # or assign just the M entries specified by the mask
        else:
            centroids = np.zeros((self.K,self.P))
            for k in range(K):
                centroids[k][mask[centroid_indices[k]]] = \
                        self.HDX_sub[centroid_indices[k]]
        return [centroids, centroid_indices]


    def initialize_centroids_kmpp(self):
        # if we use a uniform subsampling
        if self.kmpp_resample:
            mask = self.mask[0]
            X_init = self.apply_mask(self.HDX[:], mask, cross_terms = True)
        # ... otherwise we use the Sparsifier's subsampling
        else:
            mask = self.mask
            X_init = self.HDX_sub

        centroid_indices = np.zeros(self.K, dtype = int)
        centroids = np.zeros((self.K, self.P))

        # pick the first one at random from the data
        centroid_indices[0] = np.random.choice(self.N)
        # ... loading the full datapoint, or ...
        if self.full_init:
            centroids[0] = self.HDX[centroid_indices[0]]
        # ... use the masked one
        else:
            centroids[0][self.mask[centroid_indices[0]]] = \
                self.HDX_sub[centroid_indices[0]]

        # initialize the previous distance counter to max float
        # (so it's guaranteed to be overwritten in the loop)
        d_prev = np.ones(self.N) * float_info.max

        # now pick the remaining k-1 centroids
        for k in range(1,self.K):
            # distance from all the data points to the last centroid added
            d_curr = self.pairwise_distances(X_init, Y = centroids[k-1], mask = mask, 
                     transform_Y = "R")[:,0]
            # ||x - U|| is either this distance or the current minimum
            d_curr = np.min((d_prev, d_curr), axis = 0)
            # overwrite previous distances with new closest
            d_prev = np.copy(d_curr)

            # compute this to normalize d_curr_sum into a prob density, and
            # also for the check used below
            d_curr_sum = d_curr.sum()

            # if the mask didn't obliterate all distance information, then
            # pick a datapoint at random with prob proportional to its distance
            # from the current centroid set
            if d_curr_sum > 0:
                centroid_indices[k] = np.random.choice(self.N, p = d_curr/d_curr_sum)
            else:
                # then the mask obliterated all distance information, so just
                # pick one uniformly at random that's not already been chosen
                available_indices = set(range(self.N)).difference(set(centroid_indices))
                centroid_indices[k] = np.random.choice(list(available_indices))
            # finally, assign the centroid, either by setting all P entires 
            # from the dense HDX ...
            if self.full_init:
                centroids[k] = self.HDX[centroid_indices[k]]
            # ... or by setting only M entries from the sparse HDX_sub
            else:
                centroids[k][self.mask[centroid_indices[k]]] = \
                    self.HDX_sub[centroid_indices[k]]


        return [centroids, centroid_indices]


    def __init__(self, n_clusters = 8, init = 'k-means++', full_init = True,
                 kmpp_resample = True, n_init = 10,
                 max_iter = 300, n_passes = 1, **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.K = n_clusters
        self.init = init
        self.full_init = full_init
        self.kmpp_resample = kmpp_resample
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_passes = n_passes
