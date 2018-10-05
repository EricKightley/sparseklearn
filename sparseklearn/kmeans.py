import numpy as np
from sys import float_info
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
    """ Sparsified K-Means clustering.

    Parameters
    ----------

    n_clusters : int, default: 8
        The number of clusters.

    init : {ndarray, 'k-means++', 'random'}, default: 'k-means++'
        Initialization method:

        ndarray : shape (n_clusters, P). Initial cluster centers, must be 
        transformed already. 

        'k-means++': picks initial cluster centers from the data with
        probability proportional to the distance of each datapoint to the
        current initial means. More expensive but better convergence.
        These will be drawn from HDX if the sparsifier has access to it,
        otherwise they come from RHDX.

        'random': picks iniitial cluster centers uniformly at random from
        the datapoints.These will be drawn from HDX if the sparsifier has access 
        to it, otherwise they come from RHDX.

    n_init : int, default: 10
        Number of times to run k-means on new initializations. The best results
        are kept.

    max_iter : int, default: 300
        Maximum number of iterations for each run.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia for convergence.

    Attributes
    ----------

    cluster_centers_ : nd.array, shape (n_clusters, P)
        Coordinates of cluster centers

    labels_ : np.array, shape (N,)
        Labels of each point

    intertia_ : float
        Sum of squared distances of samples to their cluster center.

    """
   
    def fit(self, X = None, HDX = None, RHDX = None):
        """ Compute k-means clustering and assign labels to datapoints.
        At least one of the parameters must be set.

        Parameters
        ----------

        X : nd.array, shape (N, P), optional
            defaults to None. Dense, raw data.

        HDX : nd.array, shape (N, P), optional
            defaults to None. Dense, transformed data.

        RHDX : nd.array, shape (N, Q), optional
            defaults to None. Subsampled, transformed data.
         
        """
        self.fit_sparsifier(X=X, HDX=HDX, RHDX=RHDX)
        best_inertia = float_info.max

        for i in range(self.n_init):
            self._fit_single_trial()
            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                cluster_centers_ = self.cluster_centers_
                labels_ = self.labels_
                best_counter = self.counter

            self.cluster_centers_ =cluster_centers_
            self.labels_ = labels_
            self.inertia_ = best_inertia
            self.counter = best_counter
       
        #self.cluster_centers_ = self._reconstruct_cluster_centers(self.n_passes)

        # set how many of each example belong to each mean
        self.n_per_cluster = np.zeros(self.n_clusters, dtype = int)
        for k in range(self.n_clusters):
            self.n_per_cluster[k] = sum(self.labels_==k)


    #def _reconstruct_cluster_centers(self, n_passes):
    #    if n_passes == 1 and self.transform is not None:
    #        cluster_centers_ = self.invert_HD(self.cluster_centers_)
    #    elif n_passes == 2:
    #        cluster_centers_ = np.zeros_like(self.cluster_centers_)
    #        for k in range(self.n_clusters):
    #            cluster_members = np.where(self.labels_ == k)
    #            cluster_centers_[k] = np.mean(self.X[cluster_members], axis = 0)
    #    else:
    #        cluster_centers_ = self.cluster_centers_
    #    return cluster_centers_

    # Initialization functions

    def _initialize_cluster_centers(self):
        """ Initialize the cluster guesses. """
        if type(self.init) is np.ndarray:
            self.cluster_centers_ = self.init
            cluster_indices = []
        elif self.init == 'k-means++':
            self.cluster_centers_, cluster_indices = self._initialize_cluster_centers_kmpp()
        elif self.init == 'random':
            self.cluster_centers_, cluster_indices = self._initialize_cluster_centers_random()
        else:
            raise Exception('Initialization must be \'k-means++\', ' + 
                    '\'random\', or an np.array of initial cluster_centers')
        self.cluster_centers_ = self.cluster_centers_
        self.cluster_indices = cluster_indices
        self.labels_, self.inertia_ = self._compute_labels()
   
    def _initialize_cluster_centers_random(self):
        """ Initialize the cluster centers with K random datapoints."""
        cluster_centers_, cluster_indices = self._pick_K_datapoints(self.n_clusters)
        return [cluster_centers_, cluster_indices]


    def _initialize_cluster_centers_kmpp(self):
        """ Initialize the cluster centers using the k-means++ algorithm."""

        rng = self.check_random_state(self.random_state)
        cluster_indices = np.zeros(self.n_clusters, dtype = int)
        self.cluster_centers_ = np.zeros((self.n_clusters, self.num_feat_full))

        # pick the first one at random from the data
        cluster_indices[0] = rng.choice(self.num_samp)
        # ... loading the full datapoint, or ...
        if self.HDX is not None:
            self.cluster_centers_[0] = self.HDX[cluster_indices[0]]
        # ... using the masked one
        else:
            self.cluster_centers_[0][self.mask[cluster_indices[0]]] = \
                self.RHDX[cluster_indices[0]]

        # initialize the previous distance counter to max float
        # (so it's guaranteed to be overwritten in the loop)
        d_prev = np.ones(self.num_samp) * float_info.max

        # now pick the remaining k-1 cluster_centers
        for k in range(1,self.n_clusters):
            # squared distance from all the data points to the last cluster added
            latest_cluster = self.cluster_centers_[k-1,np.newaxis]
            d_curr = self.pairwise_distances(Y = latest_cluster)[:,0]**2
            # ||x - U|| is either this distance or the current minimum
            # overwrite current distances where we haven't improved
            where_we_have_not_improved = np.where(d_curr > d_prev)[0]
            #if where_we_have_not_improved:
            d_curr[where_we_have_not_improved] = d_prev[where_we_have_not_improved]
            d_prev = np.copy(d_curr)

            # compute this to normalize d_curr_sum into a prob density, and
            # also for the check used below
            d_curr_sum = d_curr.sum()

            # if the mask didn't obliterate all distance information, then
            # pick a datapoint at random with prob proportional to its distance
            # from the current cluster set
            if d_curr_sum > 0:
                cluster_indices[k] = rng.choice(self.num_samp, p = d_curr/d_curr_sum)
            else:
                # then the mask obliterated all distance information, so just
                # pick one uniformly at random that's not already been chosen
                available_indices = set(range(self.num_samp)).difference(set(cluster_indices))
                cluster_indices[k] = np.random.choice(list(available_indices))
            # finally, assign the cluster, either by setting all P entires 
            # from the dense HDX ...
            if self.HDX is not None:
                self.cluster_centers_[k] = self.HDX[cluster_indices[k]]
            # ... or by setting only M entries from the sparse RHDX
            else:
                self.cluster_centers_[k][self.mask[cluster_indices[k]]] = \
                    self.RHDX[cluster_indices[k]]


        return [self.cluster_centers_, cluster_indices]


    # Core algorithms

    def _compute_labels(self):
        """ Compute the labels of each datapoint."""
        d = self.pairwise_distances(Y=self.cluster_centers_)
        labels_ = d.argsort(axis=1)[:,0]
        inertia_ = np.sum([d[n,labels_[n]] for n in range(self.num_samp)])
        return [labels_, inertia_]

    def _compute_cluster_centers(self):
        """ Compute the means of each cluster."""
        #TODO: replace this with call to C function
        cluster_centers_ = np.zeros((self.n_clusters, self.num_feat_full), dtype = np.float64)
        counters = np.zeros_like(cluster_centers_, dtype = int)
        for n in range(self.num_samp):
            x = self.RHDX[n]
            l = self.labels_[n]
            cluster_centers_[l][self.mask[n]] += x
            counters[l][self.mask[n]] += 1
        for k in range(self.n_clusters):
            nonzeros = np.where(cluster_centers_[k]!=0)[0]
            cluster_centers_[k][nonzeros] *= 1 / counters[k][nonzeros]
        return cluster_centers_

    def _fit_single_trial(self):
        """ Initialize and run a single trial."""
        self._initialize_cluster_centers()
        current_iter = 0
        inertia_change = 2*self.tol
        while(current_iter < self.max_iter and inertia_change > self.tol):
            self.cluster_centers_ = self._compute_cluster_centers()
            previous_inertia = self.inertia_
            self.labels_, self.inertia_ = self._compute_labels()
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

        self.counter = current_iter

    def __init__(self, n_clusters = 8, init = 'k-means++', tol = 1e-4, 
                 n_init = 10, n_passes = 1, max_iter = 300, **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_passes = n_passes
