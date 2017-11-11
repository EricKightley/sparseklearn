
import numpy as np
from sys import float_info
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
   
    def fit(self, X, y = None):
        # y included for compatibility with sklearn.cluster.KMeans,
        # which also doesn't use it for anything...
        self.initialize(X)
        best_inertia = float_info.max

        for i in range(self.n_init):
            self.fit_single_trial()
            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                centroids = self.centroids
                labels_ = self.labels_
            self.cluster_centers_ = centroids
            self.labels_ = labels_
            self.inertia_ = best_inertia


    # Initialization functions

    def initialize_centroids(self):
        """ Initialize the centroid guesses. """
        if self.init == 'k-means++':
            centroids, centroid_indices = self.initialize_centroids_kmpp()
        elif self.init == 'random':
            centroids, centroid_indices = self.initialize_centroids_random()
        elif type(self.init) is np.ndarray:
            centroids = self.init
            centroid_indices = []
        else:
            raise Exception('Initialization must be \'k-means++\', ' + 
                    '\'random\', or an np.array of initial centroids')
        self.centroids = centroids
        self.centroid_indices = centroid_indices
        self.labels_, self.inertia_ = self.compute_labels()
   
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
            d_curr = self.pairwise_distances(self.HDX_sub, Y = centroids[k-1], 
                    mask = self.mask, transform_Y = "R")[:,0]
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


    # Core algorithms

    def compute_labels(self):
        d = self.pairwise_distances(self.HDX_sub, self.centroids, self.mask, 
                                    transform_Y = 'R')
        labels_ = d.argsort(axis=1)[:,0]
        inertia_ = np.sum([d[n,labels_[n]] for n in range(self.N)])
        return [labels_, inertia_]

    def compute_centroids(self):
        centroids = np.zeros((self.K, self.P))
        counters = np.zeros_like(centroids, dtype = int)
        for n in range(self.N):
            x = self.HDX_sub[n]
            l = self.labels_[n]
            centroids[l][self.mask[n]] += x
            counters[l][self.mask[n]] += 1
        for k in range(self.K):
            nonzeros = np.where(centroids[k]!=0)[0]
            centroids[k][nonzeros] *= 1 / counters[k][nonzeros]
        return centroids

    def fit_single_trial(self):
        self.initialize_centroids()
        current_iter = 0
        inertia_change = 2*self.tol
        while(current_iter < self.max_iter and inertia_change > self.tol):
            self.centroids = self.compute_centroids()
            previous_inertia = self.inertia_
            self.labels_, self.inertia_ = self.compute_labels()
            current_iter += 1
            inertia_change = np.abs(self.inertia_ - previous_inertia)/self.inertia_
        
        # assign convergence results
        self.iterations_ = current_iter
        if inertia_change < self.tol:
            self.converged = True
        else:
            self.converged = False
            


    def __init__(self, n_clusters = 8, init = 'k-means++', tol = 1e-4, 
                 full_init = True, n_init = 10, max_iter = 300, 
                 n_passes = 1, **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.K = n_clusters
        self.init = init
        self.full_init = full_init
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_passes = n_passes
        self.tol = tol
