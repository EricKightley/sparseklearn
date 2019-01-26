import numpy as np
from sys import float_info
from .sparsifier import Sparsifier

class KMeans(Sparsifier):
    """ Sparsified K-Means clustering.

    Parameters
    ----------

    n_components : int, default: 8
        The number of clusters.

    init : {ndarray, 'kmpp', 'random'}, default: 'kmpp'
        Initialization method:

        ndarray : shape (n_components, P). Initial cluster centers, must be 
        transformed already. 

        'kmpp': picks initial cluster centers from the data with
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

    cluster_centers_ : nd.array, shape (n_components, P)
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
       
        # set how many of each example belong to each mean
        self.n_per_cluster = np.zeros(self.n_components, dtype = int)
        for k in range(self.n_components):
            self.n_per_cluster[k] = sum(self.labels_==k)

    def _initialize_cluster_centers(self, init):
        """ Initialize the cluster guesses.
        if type(self.init) is np.ndarray:
            self.cluster_centers_ = self.init
            cluster_indices = []
        elif self.init == 'kmpp':
            self.cluster_centers_, self.cluster_indices = self._initialize_cluster_centers_kmpp()
        elif self.init == 'random':
            self.cluster_centers_, self.cluster_indices = self._initialize_cluster_centers_random()
        else:
            raise Exception('Initialization must be \'kmpp\', ' + 
                    '\'random\', or an np.array of initial cluster_centers')
        self.labels_, self.inertia_ = self._compute_labels()
        """

        means_init_array = self.means_init_array
        if means_init_array is None:
            if init == "kmpp":
                means, _ = self._pick_K_dense_datapoints_kmpp(self.n_components)
            elif init == "random":
                means, _ = self._pick_K_dense_datapoints_random(self.n_components)
            else:
                means = init
        elif means_init_array is not None:
            means = means_init_array[self.means_init_array_counter]
            self.means_init_array_counter += 1
        self.cluster_centers_ = means


    # Core algorithms

    def _compute_labels(self):
        """ Compute the labels of each datapoint."""
        d = self.pairwise_distances(Y=self.cluster_centers_)
        labels_ = d.argsort(axis=1)[:,0]
        inertia_ = np.sum([d[n,labels_[n]] for n in range(self.num_samp)])
        return [labels_, inertia_]

    def _compute_cluster_centers(self):
        """ Compute the means of each cluster."""
        resp = np.zeros((self.num_samp, self.n_components), dtype = float)
        resp[np.arange(self.num_samp), self.labels_] = 1
        cluster_centers_ = self.weighted_means(resp)
        return cluster_centers_


    def _fit_single_trial(self):
        """ Initialize and run a single trial."""
        self._initialize_cluster_centers(self.init)
        self.labels_, self.inertia_ = self._compute_labels()
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

    def __init__(self, n_components = 8, init = 'kmpp', tol = 1e-4, 
                 n_init = 10, n_passes = 1, max_iter = 300, 
                 means_init_array = None,
                 **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_passes = n_passes
        self.means_init_array = means_init_array
        self.means_init_array_counter = 0
