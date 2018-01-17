import numpy as np
import h5py
from scipy import sparse
from scipy.fftpack import dct, idct

class Sparsifier():
    """ Sparsifies input data. """


    # assignment functions

    def fit_sparsifier(self, data, override = False):
        # if we passed this object a sparsifier or 
        # a superclass of a sparsifier, 
        # just copy its attributes. This is probably a very bad idea.
        # This exists now because we need to fit a KMeans classifier
        # in the intialization of the GaussianMixture classifier
        if data.__class__.__bases__[0] is Sparsifier or \
            data.__class__ is Sparsifier:
            #[setattr(self,key,data.__dict__[key]) for key in 
            #        data.__dict__.keys()]
            copy_these_attributes = ['sparsifier_is_fit','N','M','P','gamma',
                'alpha','HDX_sub','HDX','X','D_indices','mask', 'use_ROS', 'apply_ROS']
            for attr in copy_these_attributes:
                setattr(self,attr,getattr(data,attr))

        # only fit it if it hasn't been fit, or if we explicitly override this
        if not self.sparsifier_is_fit or override:
            self.set_data(data)
            self.set_ROS()
            self.set_subsample()
            self.sparsifier_is_fit = True
            if self.normalize:
                print('normalizing')
                self.normalize_by_subsample()

    def set_data(self, data):
        """ Assigns the data as an attribute. data can either be a numpy ndarray
        or an h5py Dataset object. Sets the following attributes:
        
            self.X                 array or h5py Dataset, NxP
            self.N                 number of rows of X (number of datapoints)
            self.P                 number of cols of X (latent dimension)
            self.gamma             compression factor (recomputed)
            self.M                 size of reduced latent dimension
        """

        if not ((type(data) is h5py._hl.dataset.Dataset) or (type(data) is np.ndarray)):
            raise Exception('Data must either be an hdf5 dataset or a numpy array.')

        self.N, self.P = data.shape
        self.X = data

        ## compression factor gamma and common subsampling ratio alpha
        # if gamma is given as a float, find the number of dimensions
        if type(self.gamma) is float:
            self.M = int(np.floor(self.P * self.gamma))
        # if gamma is given as the num dimensions, find the ratio
        if type(self.gamma) is int or type(self.gamma) is np.int64:
            self.M = self.gamma
        # compute number of shared and random subsampling
        if type(self.alpha) is float:
            # then alpha is the ratio of shared indices
            Ms = int(np.floor(self.M * self.alpha))
        elif type(self.alpha) is int or type(self.alpha) is np.int64:
            # then alpha is the number of shared indices
            if self.alpha > self.M:
                raise Exception("Number of common subsamples" +
                    "alpha = {}".format(self.alpha) +
                    "exceeds total number of subsamples M = {}".format(self.M))
            else:
                Ms = self.alpha
       
        self.Ms = Ms
        self.Mr = self.M - Ms
        # overwrite alpha and gamma (either because they were ints or to 
        # account for rounding from floor function above)
        self.gamma = self.M/self.P
        self.alpha = self.Ms/self.M
        if self.verbose:
            print('Latent dimension will be reduced to',
            '{} ({} shared) from {}'.format(self.M, self.Ms, self.P), 
            'for a compression factor of',
            '{:.5} (alpha of {:.5}).'.format(self.gamma, self.alpha))


    def set_ROS(self, D_indices = None):
        """ Assigns the ROS and indices."""
        if self.use_ROS:
            # if we're told to compute it, do so
            if self.compute_ROS == True:
                self.D_indices = np.array([i for i in range(self.P) 
                    if np.random.choice([0,1])])
                HDX = self.apply_ROS(self.X[:])
            # otherwise load it
            elif self.fROS != None:
                HDX, D_indices = self.set_ROS_from_input(self.fROS, D_indices)
                self.D_indices = D_indices

        else:
            # if we're not using the ROS just set HDX to be X
            HDX = self.X[:].astype(float)
            self.D_indices = np.ones(self.N)


        self.HDX = HDX


    def set_subsample(self):
        """ Assign the subsampled data once the ROS has been applied. Needs
        to be updated to work with C functions. 
        """
        self.mask, self.shared = self.generate_mask(self.P, self.Ms, self.Mr, self.N)
        self.HDX_sub = self.apply_mask(self.HDX, self.mask)

    # ROS functions
    def apply_ROS(self, X):
        # copy it for now 
        Y = np.copy(X)
        # apply D matrix
        Y[:,self.D_indices] *= -1
        # apply H matrix
        Y = dct(Y, norm = 'ortho', axis = 1, overwrite_x = False) 
        return Y

    def invert_ROS(self, X):
        # copy it for now
        Y = np.copy(X)
        Y = idct(Y, norm = 'ortho', axis = 1, overwrite_x = False)
        Y[:,self.D_indices] *= -1
        return Y

    def set_ROS_from_input(self, fROS, D_indices):
        if type(fROS) is h5py._hl.dataset.Dataset:
            HDX = fROS['HDX']
            D_indices = fROS['D_indices']
        elif type(data) is np.ndarray:
            HDX = HDX
        return [HDX, D_indices]

    def write_ROS(self, fROS, HDX, D_indices):
        """ Writes HDX and D_indices to file fROS (D_indices are needed to
        invert the transformation)."""
        
        if 'HDX' in fROS:
            if self.verbose:
                print('Deleting existing ROS dataset in hdf5 file {}'.format(fROS))
            del fROS['HDX']
        if 'D_indices' in self.fROS:
            if self.verbose:
                print('Deleting existing D_indices dataset in hdf5 file {}'.format(fROS))
            del fROS['D_indices']
        if self.verbose:
            print('Writing ROS and D_indices to hdf5 file {}'.format(fROS))
        fROS.create_dataset('HDX', data = HDX, dtype = 'd')
        fROS.create_dataset('D_indices', data = D_indices, dtype = 'int')


    def normalize_by_subsample(self):
        normalizer = np.mean(np.linalg.norm(self.HDX_sub,axis=1,ord=2))
        self.HDX /= normalizer
        self.HDX_sub /= normalizer
        self.normalizer = normalizer


    # masking functions

    def generate_mask(self, P, Ms, Mr, N):
        """
        """
        inds = [p for p in range(P)]
        np.random.shuffle(inds)
        shared_mask = inds[:Ms]
        inds = inds[Ms:]

        random_masks = [np.random.choice(inds, Mr, replace=False)
                        for n in range(N)]

        mask = np.concatenate((random_masks, 
               np.tile(shared_mask, (N,1)).astype(int)), axis = 1)
        mask.sort(axis=1)

        return [mask, np.sort(shared_mask)]

    def invert_mask_bool(self):
        """ Returns P by N binary sparse matrix. Each row indicates which
        of the N data points has the pth dimension preserved in the mask.
        """
        col_inds = [n for n in range(self.N) for m in range(self.M)]
        row_inds = list(self.mask.flatten())
        data = np.ones_like(row_inds)
        mask_binary = sparse.csr_matrix( (data, (row_inds, col_inds)), 
                      shape = (self.P,self.N), dtype = bool)
        return mask_binary

    def apply_mask(self, X, mask):
        """ Apply mask to X. 
        """
        if X.ndim == 2:
            if X.shape[0] != mask.shape[0]:
                raise Exception('Number of rows in mask must agree with number',
                        'of rows in X')
            X_masked = np.array([X[n][mask[n]] for n in range(mask.shape[0])])
        elif X.ndim == 1:
            X_masked = np.array([X[mask[n]] for n in range(mask.shape[0])])
        else:
            raise Exception('X must be 1 or 2-dimensional')
        return X_masked


    # masked matrix operations

    def polynomial_combination(self, W, power = 1):
        """

        Computes sum_n (w[n,k] * (x[n] * iota[n] )**power

        W is self.N by K, so that the output has K rows, each of which
        is a sum above using a column of w. This seems backwards. 

        """
        if W.ndim != 2:
            raise Exception('W must be a 2D array')
        _, K = np.shape(W)
        comb = np.zeros((K, self.P))

        if power == 1:
            for k in range(K):
                for n in range(self.N):
                    comb[k][self.mask[n]] += W[n,k] * self.HDX_sub[n]
        else:
            for k in range(K):
                for n in range(self.N):
                    comb[k][self.mask[n]] += W[n,k] * self.HDX_sub[n]**power

        return comb


    def pairwise_distances(self, Y = None, W = None, 
            transform_Y = "R", transform_W = "R"):

        X = self.HDX_sub

        # transform Y if we need to
        if "HD" in transform_Y and Y is not None:
            Y = self.apply_ROS(Y)
        # assign Y to be X if we need to
        elif Y is None:
            Y = X
        # augment Y if we need to
        if Y.ndim == 1:
            Y = Y[np.newaxis,:]
        
        # transform W if we need to
        if "HD" in transform_W and W is not None:
            W = self.apply_ROS(W)

        # set up the distances output array
        K, _ = np.shape(Y)
        dist = np.zeros((K, self.N))

        # don't check this every time we go through the loop
        if "R" in transform_Y:
            subsample_Y = True
        else:
            subsample_Y = False

        if "R" in transform_W:
            subsample_W = True
        else:
            subsample_W = False

        # compute the distances

        # ... in two cases. First, if no weights are given
        if W is None:
            for k in range(K):
                if subsample_Y:
                    y = self.apply_mask(Y[k], self.mask)
                else:
                    y = Y[k]
                dist[k] = np.linalg.norm(y - X, axis = 1)
        # ... or alternatively if they are given
        else:
            for k in range(K):
                if subsample_Y:
                    y = self.apply_mask(Y[k], self.mask)
                else:
                    y = Y[k]
                if subsample_W:
                    w = self.apply_mask(W[k], self.mask)
                else:
                    w = W[k]
                Xmys = (X-y)**2
                dist[k] = np.sqrt([np.dot(Xmys[n], w[n]) for n in range(self.N)])
        return dist.T

    def pick_K_datapoints(self, K):
        # pick K data points at random uniformly
        cluster_indices = np.random.choice(self.N, K, replace = False)
        cluster_indices.sort()
        # assign the cluster_centers as dense members of HDX ...
        if self.full_init:
            cluster_centers_ = np.array(self.HDX[cluster_indices])
        # or assign just the M entries specified by the mask
        else:
            cluster_centers_ = np.zeros((self.K,self.P))
            for k in range(K):
                cluster_centers_[k][mask[cluster_indices[k]]] = \
                        self.HDX_sub[cluster_indices[k]]
        return [cluster_centers_, cluster_indices]




    def __init__(self, gamma = 1.0, alpha = 0.0, verbose = False, fROS = None, 
                 use_ROS = True, compute_ROS = True, 
                 normalize = False):

        # assign constants
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.fROS = fROS
        self.use_ROS = use_ROS
        self.compute_ROS = compute_ROS
        self.sparsifier_is_fit = False
        self.normalize = normalize


