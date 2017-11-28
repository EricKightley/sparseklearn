
import numpy as np
from sys import float_info
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
   
    def fit(self, X, y = None):
        # y included for compatibility with sklearn.cluster.KMeans,
        # which also doesn't use it for anything...
        self.fit_sparsifier(X)
        best_inertia = float_info.max

        for i in range(self.n_init):
            self.fit_single_trial()
            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                cluster_centers_ = self.cluster_centers_
                labels_ = self.labels_
            self.cluster_centers_ =cluster_centers_
            self.labels_ = labels_
            self.inertia_ = best_inertia
       
        # postprocessing

        self.cluster_centers_ = self.reconstruct_cluster_centers(self.n_passes)


        # set how many of each example belong to each mean
        self.n_per_cluster = np.zeros(self.n_clusters, dtype = int)
        for k in range(self.n_clusters):
            self.n_per_cluster[k] = sum(self.labels_==k)


    def reconstruct_cluster_centers(self, n_passes):
        if n_passes == 1 and self.use_ROS:
            cluster_centers_ = self.invert_ROS(self.cluster_centers_, self.D_indices)
        elif n_passes == 2:
            cluster_centers_ = np.zeros_like(self.cluster_centers_)
            for k in range(self.K):
                cluster_members = np.where(self.labels_ == k)
                cluster_centers_[k] = np.mean(self.X[cluster_members], axis = 0)
        return cluster_centers_

    # Initialization functions

    def initialize_cluster_centers(self):
        """ Initialize the cluster guesses. """
        if self.init == 'k-means++':
            self.cluster_centers_, cluster_indices = self.initialize_cluster_centers_kmpp()
        elif self.init == 'random':
            self.cluster_centers_, cluster_indices = self.initialize_cluster_centers_random()
        elif type(self.init) is np.ndarray:
            self.cluster_centers_ = self.init
            cluster_indices = []
        else:
            raise Exception('Initialization must be \'k-means++\', ' + 
                    '\'random\', or an np.array of initial cluster_centers')
        self.cluster_centers_ = self.cluster_centers_
        self.cluster_indices = cluster_indices
        self.labels_, self.inertia_ = self.compute_labels()
   
    def initialize_cluster_centers_random(self):
        # pick K data points at random uniformly
        cluster_indices = np.random.choice(self.N, self.K, replace = False)
        cluster_indices.sort()
        # assign the cluster_centers as dense members of HDX ...
        if self.full_init:
            self.cluster_centers_ = np.array(self.HDX[cluster_indices])
        # or assign just the M entries specified by the mask
        else:
            self.cluster_centers_ = np.zeros((self.K,self.P))
            for k in range(K):
                self.cluster_centers_[k][mask[cluster_indices[k]]] = \
                        self.HDX_sub[cluster_indices[k]]
        return [self.cluster_centers_, cluster_indices]


    def initialize_cluster_centers_kmpp(self):

        cluster_indices = np.zeros(self.K, dtype = int)
        self.cluster_centers_ = np.zeros((self.K, self.P))

        # pick the first one at random from the data
        cluster_indices[0] = np.random.choice(self.N)
        # ... loading the full datapoint, or ...
        if self.full_init:
            self.cluster_centers_[0] = self.HDX[cluster_indices[0]]
        # ... use the masked one
        else:
            self.cluster_centers_[0][self.mask[cluster_indices[0]]] = \
                self.HDX_sub[cluster_indices[0]]

        # initialize the previous distance counter to max float
        # (so it's guaranteed to be overwritten in the loop)
        d_prev = np.ones(self.N) * float_info.max

        # now pick the remaining k-1 cluster_centers
        for k in range(1,self.K):
            # distance from all the data points to the last cluster added
            d_curr = self.pairwise_distances(Y = self.cluster_centers_[k-1])[:,0]
            # ||x - U|| is either this distance or the current minimum
            d_curr = np.min((d_prev, d_curr), axis = 0)
            # overwrite previous distances with new closest
            d_prev = np.copy(d_curr)

            # compute this to normalize d_curr_sum into a prob density, and
            # also for the check used below
            d_curr_sum = d_curr.sum()

            # if the mask didn't obliterate all distance information, then
            # pick a datapoint at random with prob proportional to its distance
            # from the current cluster set
            if d_curr_sum > 0:
                cluster_indices[k] = np.random.choice(self.N, p = d_curr/d_curr_sum)
            else:
                print("WAHHH")
                # then the mask obliterated all distance information, so just
                # pick one uniformly at random that's not already been chosen
                available_indices = set(range(self.N)).difference(set(cluster_indices))
                cluster_indices[k] = np.random.choice(list(available_indices))
            # finally, assign the cluster, either by setting all P entires 
            # from the dense HDX ...
            if self.full_init:
                self.cluster_centers_[k] = self.HDX[cluster_indices[k]]
            # ... or by setting only M entries from the sparse HDX_sub
            else:
                self.cluster_centers_[k][self.mask[cluster_indices[k]]] = \
                    self.HDX_sub[cluster_indices[k]]


        return [self.cluster_centers_, cluster_indices]


    # Core algorithms

    def compute_labels(self):
        d = self.pairwise_distances(Y=self.cluster_centers_)
        labels_ = d.argsort(axis=1)[:,0]
        inertia_ = np.sum([d[n,labels_[n]] for n in range(self.N)])
        return [labels_, inertia_]

    def compute_cluster_centers(self):
        self.cluster_centers_ = np.zeros((self.K, self.P))
        counters = np.zeros_like(self.cluster_centers_, dtype = int)
        for n in range(self.N):
            x = self.HDX_sub[n]
            l = self.labels_[n]
            self.cluster_centers_[l][self.mask[n]] += x
            counters[l][self.mask[n]] += 1
        for k in range(self.K):
            nonzeros = np.where(self.cluster_centers_[k]!=0)[0]
            self.cluster_centers_[k][nonzeros] *= 1 / counters[k][nonzeros]
        return self.cluster_centers_

    def fit_single_trial(self):
        self.initialize_cluster_centers()
        current_iter = 0
        inertia_change = 2*self.tol
        while(current_iter < self.max_iter and inertia_change > self.tol):
            self.cluster_centers_ = self.compute_cluster_centers()
            previous_inertia = self.inertia_
            self.labels_, self.inertia_ = self.compute_labels()
            current_iter += 1
            if self.inertia_ > 0:
                inertia_change = np.abs(self.inertia_ - previous_inertia)/self.inertia_
            else:
                inertia_change = 0
        
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
        # for compatibility with sklearn.cluster.KMeans
        self.n_clusters = n_clusters
        self.init = init
        self.full_init = full_init
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_passes = n_passes
        self.tol = tol
