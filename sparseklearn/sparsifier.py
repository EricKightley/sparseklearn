import numpy as np
import h5py
from scipy import sparse
from scipy.fftpack import dct, idct

class Sparsifier():


    # assignment functions

    def fit_sparsifier(self, data, override = False):
        # if we passed this object a sparsifier or 
        # a superclass of a sparsifier, 
        # just copy its attributes. This is probably a very bad idea.
        # This exists now because we need to fit a KMeans classifier
        # in the intialization of the GaussianMixture classifier
        if data.__class__.__bases__[0] is Sparsifier or \
            data.__class__ is Sparsifier:
            [setattr(self,key,data.__dict__[key]) for key in 
                    data.__dict__.keys()]

        # only fit it if it hasn't been fit, or if we explicitly override this
        if not self.sparsifier_is_fit or override:
            self.set_data(data)
            self.set_ROS()
            self.set_subsample()
            self.sparsifier_is_fit = True


    def set_data(self, data):
        """ Assigns the data as an attribute. data can either be a numpy ndarray
        or an h5py Dataset object. Sets the following attributes:
        
            self.X                 array or h5py Dataset, NxP
            self.N                 number of rows of X (number of datapoints)
            self.P                 number of cols of X (latent dimension)
            self.gamma             compression factor (recomputed)
            self.M                 size of reduced latent dimension
        """

        if type(data) is h5py._hl.dataset.Dataset:
            X_type = 'hdf5'
        elif type(data) is np.ndarray:
            X_type = 'array'
        else:
            raise Exception('Data must either be an hdf5 dataset or a numpy array.')

        self.N, self.P = data.shape
        self.X = data

        if self.verbose:
            print('Data assigned as {}, '.format(X_type) + 
                'and is {} '.format(self.N) + 'by {} '.format(self.P) +
                '(data points by latent dimension).')

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


    def set_ROS(self):
        """ Assigns the ROS and indices."""
        if self.use_ROS:
            # if we're told to compute it, do so
            if self.compute_ROS == True:
                self.D_indices = self.generate_ROS(self.P)
                HDX = self.apply_ROS(self.X[:])
                # ... but write it if we're allowed to so
                if self.write_permission == True:
                    self.write_ROS(self.fROS, HDX, self.D_indices)
            # otherwise load it
            elif self.fROS != None:
                HDX, D_indices = self.read_ROS(self.fROS)

        else:
            # if we're not using the ROS just set HDX to be X
            HDX = self.X[:].astype(float)
            self.D_indices = np.ones(self.N)
        self.HDX = HDX


    def set_subsample(self):
        """ Assign the subsampled data once the ROS has been applied. Needs
        to be updated to work with C functions. 
        """
        mask, shared = self.generate_mask(self.P, self.Ms, self.Mr, self.N)
        # the mask returns HDX_sub in compact form
        HDX_sub = self.apply_mask(self.HDX, mask)
        # convert it if we need it in dense or sparse form
        if self.sparsity_format == 'dense' or self.sparsity_format == 'sparse':
            row_inds = [i for i in range(self.N) for j in range(self.M)]
            HDX_sub = sparse.coo_matrix( ( HDX_sub.flatten(), 
                      (row_inds, list(mask.flatten())) ) , 
                      shape = (self.N,self.P))
            if self.sparsity_format == 'dense':
                HDX_sub = np.asarray(HDX_sub.todense())
        self.HDX_sub = HDX_sub
        self.mask = mask
        self.shared = shared


    # ROS functions
    def generate_ROS(self, P):
        D_indices = np.array([i for i in range(P) if np.random.choice([0,1])])
        return D_indices

    def apply_ROS(self, X):
        # copy it for now 
        X = np.copy(X)
        # apply D matrix
        X[:,self.D_indices] *= -1
        # apply H matrix
        X = dct(X, norm = 'ortho', axis = 1, overwrite_x = False) 
        return X

    def invert_ROS(self, X, D):
        # copy it for now
        X = np.copy(X)
        X = idct(X, norm = 'ortho', axis = 1, overwrite_x = False)
        X[:,D] *= -1
        return X

    def read_ROS(self, fROS):
        HDX = fROS['HDX']
        D_indices = fROS['D_indices']
        return [HDX, D_indices]

    def write_ROS(self, fROS, HDX, D_indices):
        """ Writes HDX and D_indices to file fROS (D_indices are needed to
        invert the transformation)."""
        
        if self.write_permission == False:
            raise Exception('Trying to write the ROS transform to disk but' + 
                             'write_permission is False')
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


    def pairwise_distances(self, Y = None, X = None, W = None, 
            transform_X = "", transform_Y = "R", transform_W = "R"):

        # assign X if we need to
        if X is None:
            X = self.HDX_sub
        # augment X if we need to
        if X.ndim == 1:
            X = X[np.newaxis,:]
        
        # transform and subsample X if we need to
        if "HD" in transform_X:
            X = self.apply_ROS(X)
        if "R" in transform_X:
            X = self.apply_mask(X, self.mask)

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
                #dist[k] = np.sqrt(np.dot((X-y)**2, w.T))
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
                 write_permission = False, use_ROS = True, compute_ROS = True, 
                 sparsity_format = 'compact'):

        # assign constants
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.fROS = fROS
        self.write_permission = write_permission
        self.use_ROS = use_ROS
        self.compute_ROS = compute_ROS
        self.sparsity_format = sparsity_format
        self.sparsifier_is_fit = False


