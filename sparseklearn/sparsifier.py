import numpy as np
import h5py
from scipy import sparse
from scipy.fftpack import dct, idct
from sklearn.metrics.pairwise import pairwise_distances as skpd

class Sparsifier():


    # assignment functions

    def initialize(self, data):
        self.set_data(data)
        self.set_ROS()
        self.set_subsample()

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

        # compute compression factor
        # if gamma is given as a float, find M
        if type(self.gamma) is float:
            self.M = int(np.floor(self.P * self.gamma))
        # if gamma is given as an integer, find gamma
        if type(self.gamma) is int or type(self.gamma) is np.int64:
            self.M = self.gamma
        # overwrite gamma (either because it was an int or to account for
        # rounding from floor function above)
        self.gamma = self.M/self.P
        if self.verbose:
            print('Latent dimension will be reduced to {} from'.format(self.M),
                '{} for a compression factor of {}.'.format(self.P, self.gamma))

    def set_ROS(self):
        """ Assigns the ROS and indices."""
        if self.use_ROS:
            # if we're told to compute it, do so
            if self.compute_ROS == True:
                D_indices = self.generate_ROS(self.P)
                HDX = self.apply_ROS(self.X[:], D_indices)
                # ... but write it if we're allowed to so
                if self.write_permission == True:
                    self.write_ROS(self.fROS, HDX, D_indices)
            # otherwise load it
            elif self.fROS != None:
                HDX, D_indices = self.read_ROS(self.fROS)

        else:
            # if we're not using the ROS just set HDX to be X
            HDX = self.X[:].astype(float)
            D_indices = []
        self.HDX, self.D_indices = HDX, D_indices


    def set_subsample(self):
        """ Assign the subsampled data once the ROS has been applied. Needs
        to be updated to work with C functions. 
        """
        if self.constant_subsample:
            N = 1
        else:
            N = self.N
        mask = self.generate_mask(self.P, self.M, N)
        HDX_sub = self.apply_mask(self.HDX, mask)
        if self.dense_subsample:
            row_inds = [i for i in range(self.N) for j in range(self.M)]
            HDX_sub = sparse.coo_matrix( ( HDX_sub.flatten(), 
                      (row_inds, list(mask.flatten())) ) , 
                      shape = (self.N,self.P))
            HDX_sub = np.asarray(HDX_sub.todense())
        self.HDX_sub = HDX_sub
        self.mask = mask


    # ROS functions
    def generate_ROS(self, P):
        D_indices = np.array([i for i in range(P) if np.random.choice([0,1])])
        return D_indices

    def apply_ROS(self, X, D):
        # copy it for now 
        X = np.copy(X)
        # apply D matrix
        X[:,D] *= -1
        # apply H matrix
        X = dct(X, norm = 'ortho', axis = 1, overwrite_x = False) 
        return X

    def ROS_test(self, X_test):
        """ Apply the ROS used on the dense dataset to another test dataset. """
        X_test[:,self.D_indices] *= -1
        X_test = dct(X_test, norm = 'ortho', axis = 1, overwrite_x = False) 
        return X_test

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

    def generate_mask(self, P, M, N):
        """ Given an n x p array ::X (each row is a datapoint of dimension p) and an integer
        ::m <= p, (uniformly) randomly keeps m entries for each of the n datapoints. If
        ::self.constant_subsample then each row has the same columns extracted. This latter
        method is currently inefficient and is for comparison purposes only. 
        """

        if N == 1:
            mask = np.sort(np.random.choice(P,M,replace=False))
        else:
            mask = np.array([np.sort(np.random.choice(
                                 P, M, replace = False))
                                 for i in range(N)])

        return mask

    def apply_mask(self, X, mask, cross_terms = False):
        """ Apply mask to X. 
        """
        if X.ndim == 1:
            X = X[np.newaxis,:]
        if mask.ndim ==1:
            mask = mask[np.newaxis,:]
        N,P = X.shape
        K,M = mask.shape

        if cross_terms:
            X_masked = np.array([X[n][mask[k]] for n in range(N) for k in range(K)])
        else:
            if N != K:
                raise Exception('If no cross-terms are desired, number of mask' + \
                'rows must equal number of X rows')
            X_masked = np.array([X[n][mask[n]] for n in range(N)])
        return X_masked

    def pairwise_distances(self, X, Y = None, mask = None, D = None, 
                           transform_X = "", transform_Y = ""):
        """
        """
        # perform some error checks
        if ("HD" in transform_X or "HD" in transform_Y) and type(D) == type(None):
            raise Exception("Cannot apply ROS without indices D")
        if "HD" in transform_X:
            X = self.apply_ROS(X, D)
        if "R" in transform_X:
            X = self.apply_mask(X, mask,cross_terms = False)
        if "HD" in transform_Y:
            Y = self.apply_ROS(Y, D)

        K = np.shape(Y)[0]
        N = np.shape(X)[0]

        dist = np.zeros((K,N))
        for k in range(K):
            if "R" in transform_Y:
                y = self.apply_mask(Y[k], mask, cross_terms = True)
            else:
                y = Y[k]
            dist[k] = np.linalg.norm(y - X, axis = 1)
        #dist = skpd(X,Y)
        #dist = [np.linalg.norm(X - Y[k], axis = 1) for k in range(Y.shape[0])]
        return dist.T



    def __init__(self, gamma = 1.0, verbose = False, fROS = None, write_permission = False,
                 use_ROS = True, compute_ROS = True, 
                 dense_subsample = False, constant_subsample = False):

        # assign constants
        self.gamma = gamma
        self.verbose = verbose
        self.fROS = fROS
        self.write_permission = write_permission
        self.use_ROS = use_ROS
        self.compute_ROS = compute_ROS
        self.dense_subsample = dense_subsample
        self.constant_subsample = constant_subsample



