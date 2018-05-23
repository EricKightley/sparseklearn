import numpy as np
import h5py
import os
import numbers
from scipy import sparse
from scipy.fftpack import dct, idct
from numpy.ctypeslib import ndpointer
import ctypes as ct

from .fastLA import pairwise_l2_distances_with_self
from .fastLA import pairwise_l2_distances_with_full
from .fastLA import compute_weighted_first_moment_array
from .fastLA import compute_weighted_first_and_second_moment_array

from .fastLA import pairwise_mahalanobis_distances_spherical
from .fastLA import pairwise_mahalanobis_distances_diagonal

class Sparsifier():
    """ Sparsifier.

    Compresses data through sparsification. Permits several operations on
    sparisified data. 
    
    Parameters
    ----------

    num_feat_full : int
        Dimension of a full sample.

    num_feat_comp : int
        The number of dimensions to keep in the compressed data. 

    num_samp : int
        The number of samples in the dataset. 

    transform : {'dct', None}, defaults to 'dct'.
        The preconditioning transform.
        Determines what form of H to use in the preconditioning transform HD. 
        Any method other than None will also use the diagonal D matrix (which 
        can be set using the D_indices parameter). Must be one of::
            
            'dct'  discrete cosine transform
            None no transform

    mask : nd.array, shape (n_datapoints, dim_mask), optional
        defaults to None. The user-provided mask. If None, mask is
        generated using the generate_mask method.

    num_feat_shared : int, defaults to 0.
        The minimum number of dimensions to be shared across all samples in the
        compressed data.

    D_indices : nd.array, shape (n_datapoints,), optional
        defaults to None. The user-provided diagonal of the preconditioning matrix D. 
        If None, generated using the generate_D_indices method. 


    Attributes
    ----------

    mask : nd.array, shape (num_samp, num_feat_comp)
        The mask used to sparsify the data. Array of integers, each row is the
        indices specifying which entries that sample were kept. 

    D_indices : nd.array, shape (n_signflips,)
        Defines the preconditioning matrix D. Array of integers, 
        the indices of the preconditioning matrix D with sign -1.

    """

    ###########################################################################
    # Temp - set random seed

    def check_random_state(self, seed):
        """Turn seed into a np.random.RandomState instance
        Parameters
        ----------
        seed : None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)

    ###########################################################################
    # Preconditoning and mask generation

    def _generate_D_indices(self):
        """ Randomly generate the D matrix in the HD transform. Store only the 
        indices where D == -1. """
        rng = self.check_random_state(self.random_state)
        if self.transform in ['dct']:
            D_indices = np.array([i for i in range(self.num_feat_full) 
                if rng.choice([0,1])])
        elif self.transform is None:
            D_indices = np.array([])
        return D_indices

    def _generate_mask(self):
        """Generate a sparsifying mask."""
        rng = self.check_random_state(self.random_state)
        # pick the shared indices
        all_indices = list(range(self.num_feat_full))
        rng.shuffle(all_indices)
        shared_mask = all_indices[:self.num_feat_shared]
        # pick what's left randomly
        remaining_indices = all_indices[self.num_feat_shared:]
        num_left_to_draw = self.num_feat_comp - self.num_feat_shared
        if num_left_to_draw > 0:
            random_masks = [rng.choice(remaining_indices, 
                num_left_to_draw, replace=False) for n in range(self.num_samp)]
            mask = np.concatenate((random_masks, 
                np.tile(shared_mask, (self.num_samp,1)).astype(int)), axis = 1)
        else:
            # then all dimensions are shared
            mask = np.tile(shared_mask, (self.num_samp,1)).astype(int)

        mask.sort(axis=1)
        return mask

    ###########################################################################
    # Transform and masking tools

    def apply_mask(self, X):
        """ Apply the mask to X.

        Parameters
        ----------

        X : nd.array, shape(n, P)


        Returns
        -------

        RX : nd.array, shape(n, Q)
            Masked X. The nth row of RX is X[n][mask[n]].

        """
        if X.ndim == 2:
            if X.shape[0] != self.mask.shape[0]:
                raise Exception('Number of rows in mask must agree with number',
                        'of rows in X')
            X_masked = np.array([X[n][self.mask[n]] for n in range(self.mask.shape[0])])
        elif X.ndim == 1:
            X_masked = np.array([X[self.mask[n]] for n in range(self.mask.shape[0])])
        else:
            raise Exception('X must be 1 or 2-dimensional')
        return X_masked

    def apply_HD(self, X):
        """ Apply the preconditioning transform to X. 

        Parameters
        ----------

        X : nd.array, shape (n, P)
            The data to precondition. Each row is a datapoint. 


        Returns
        -------

        HDX : nd.array, shape (n, P)
            The transformed data. 

        """
        # copy it for now 
        Y = np.copy(X)
        # apply D matrix
        Y[:,self.D_indices] *= -1
        # apply H matrix
        Y = dct(Y, norm = 'ortho', axis = 1, overwrite_x = False) 
        return Y

    def invert_HD(self, HDX):
        """ Apply the inverse of HD to HDX. 

        Parameters
        ----------

        HDX : nd.array, shape (n, P)
            The preconditioned data. Each row is a datapoint. 


        Returns
        -------

        X : nd.array, shape (n, P)
            The raw data. 

        """
        X = np.copy(HDX)
        X = idct(X, norm = 'ortho', axis = 1, overwrite_x = False)
        X[:,self.D_indices] *= -1
        return X

    def invert_mask_bool(self):
        """ Compute the mask inverse.

        The mask is an array indicating which dimensions are kept for each
        data point. The inverse mask is an array indicating which datapoints
        keep this dimension, for each dimension. For computational efficiency,
        the inverse mask is given as a sparse boolean array whereas the mask
        is a (smaller) dense integer array. 

        Returns
        -------

        mask_inverse : sparse.csr_matrix, bool, shape (P,N)
            The mask inverse. The ij entry is 1 if the jth datapoint
            keeps the ith dimension under the mask, and 0 otherwise;
            in other words, 1 if i is in the list mask[j].
            
        """
        col_inds = [n for n in range(self.num_samp) for m in range(self.num_feat_comp)]
        row_inds = list(self.mask.flatten())
        data = np.ones_like(row_inds)
        mask_binary = sparse.csr_matrix( (data, (row_inds, col_inds)), 
                      shape = (self.num_feat_full,self.num_samp), dtype = bool)
        return mask_binary


    ###########################################################################
    # Fitting

    def _set_HDX(self, transform, X, HDX, RHDX):
        """ Wrapper to compute HDX from X or assign it if user-specified. """
        if HDX is not None:
            return HDX
        elif (X is not None and transform == 'dct'):
            return self.apply_HD(X)
        elif (X is not None and transform is None):
            return X

    def _set_RHDX(self, X, HDX, RHDX):
        """ Wrapper to compute RHDX from HDX or assign it if user-specified. """
        if RHDX is not None:
            return RHDX.astype(float)
        elif HDX is not None:
            return self.apply_mask(HDX.astype(float))
        else:
            return self.apply_mask(X.astype(float))

    def fit_sparsifier(self, X = None, HDX = None, RHDX = None):
        """ Fit the sparsifier to specified data. 
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

        # set D_indices
        # self.D_indices = self._set_D(self.transform, self.D_indices, self.num_feat_full)
        # set mask
        # self.mask = self._set_mask(self.mask, self.num_feat_full, self.num_feat_comps, self.num_feat_compr, self.num_samp) 
        # compute HDX and RHDX
        HDX = self._set_HDX(self.transform, X, HDX, RHDX)
        RHDX = self._set_RHDX(X, HDX, RHDX)
        # assign data
        self.X = X
        self.HDX = HDX
        self.RHDX = RHDX
        
    ###########################################################################
    # Operations on masked data

    def pairwise_distances(self, Y = None):
        """ Computes the pairwise distance between each sparsified sample,
        or between each sparsified sample and each full sample in Y if 
        Y is given. 

        Parameters
        ------

        Y : nd.array, shape (K, P), optional
            defaults to None. Full, transformed samples.

        Returns
        -------

        distances : nd.array, shape(K or N, N)
            distances between each pair of samples (if Y is None) or distances
            between each sample and each row in Y. 

        """

        if Y is None:
            result = np.zeros((self.num_samp, self.num_samp), dtype=np.float64)
            pairwise_l2_distances_with_self(result, self.RHDX, self.mask, 
                self.num_samp, self.num_feat_comp, self.num_feat_full)
        else:
            K = Y.shape[0]
            result = np.zeros((self.num_samp, K), dtype = np.float64)
            pairwise_l2_distances_with_full(result, self.RHDX, Y, self.mask, 
                self.num_samp, K, self.num_feat_comp, self.num_feat_full)

        return result

    def weighted_means(self, W):
        """ Computes weighted full means of sparsified samples. Currently this 
        is also used to compute hard assignments but should be updated for 
        speed later - zeros in W are multiplied through. 

        Parameters
        ------

        W : nd.array, shape (N, K)
            Weights. Each row corresponds to a sample, each column to a set of
            weights. The columns of W should sum to 1. There is no necessary 
            correspondence between the columns of W.

        Returns
        -------

        means : nd.array, shape (K,P)
            Weighted full means. Each row corresponds to a possible independent
            set of weights (for example, a binary W with K columns would give
            the means of K clusters). 
        """

        K = np.shape(W)[1]
        means = np.zeros((K, self.num_feat_full), dtype = np.float64)
        compute_weighted_first_moment_array(
                               means,
                               self.RHDX,
                               self.mask,
                               W,
                               self.num_samp,
                               K,
                               self.num_feat_comp,
                               self.num_feat_full)
        return means

    def weighted_means_and_variances(self, W):
        """ Computes weighted full means and variances of sparsified samples. 
        Currently also used to compute hard assignments but should be updated 
        for speed later - zeros in W are multiplied through. 

        Parameters
        ------

        W : nd.array, shape (N, K)
            Weights. Each row corresponds to a sample, each column to a set of
            weights. The columns of W should sum to 1. There is no necessary 
            correspondence between the columns of W.

        Returns
        -------

        means : nd.array, shape (K,P)
            Weighted full means. Each row corresponds to a possible independent
            set of weights (for example, a binary W with K columns would give
            the means of K clusters). 

        variances : nd.array, shape (K,P)
            Weighted full variances. Each row corresponds to a possible independent
            set of weights (for example, a binary W with K columns would give
            the variances of K clusters). 
        """

        K = np.shape(W)[1]
        means = np.zeros((K, self.num_feat_full), dtype = np.float64)
        second_moments = np.zeros((K, self.num_feat_full), dtype = np.float64)
        compute_weighted_first_and_second_moment_array(
                               means,
                               second_moments,
                               self.RHDX,
                               self.mask,
                               W,
                               self.num_samp,
                               K,
                               self.num_feat_comp,
                               self.num_feat_full)
        variances = second_moments - means**2
        return[means, variances]

    def pairwise_mahalanobis_distances(self, means, covariances, covariance_type):
        """ Computes the mahalanobis distance between each compressed sample and
        each full mean (each row of means).

        Parameters
        ----------

        means : nd.array, shape (K,P)
            The means with which to take the mahalanobis distances. Each row of
            ::means is a single mean in P-dimensional space. 

        covariances : nd.array, shape (K,P) or shape (P,). 
            The non-zero entries of the covariance matrix. If 
            ::covariance_type is 'spherical', must be shape (P,). If
            ::covariance_type is 'diag', must be shape (K,P)

        covariance_type : string. Must be one of 
        
            'spherical' (each component has its own single variance)
            'diag' (each component has its own diagonal covariance matrix)

        Returns
        -------

        distances : nd.array, shape (N,K)
            The pairwise mahalanobis distances. 

        """
        #TODO check that means is 2D and it has number of columns == self.num_feat_full
        #TODO check that covariances is the right shape for each case
        #TODO add a test to catch the exception
        K = means.shape[0]
        distances = np.zeros((self.num_samp, K), dtype = np.float64)
        if covariance_type == 'spherical':
            pairwise_mahalanobis_distances_spherical(distances,
                                                     self.RHDX,
                                                     means,
                                                     self.mask,
                                                     covariances,
                                                     self.num_samp,
                                                     K,
                                                     self.num_feat_comp,
                                                     self.num_feat_full)
        elif covariance_type == 'diag':
            pairwise_mahalanobis_distances_diagonal(distances,
                                                    self.RHDX,
                                                    means,
                                                    self.mask,
                                                    covariances,
                                                    self.num_samp,
                                                    K,
                                                    self.num_feat_comp,
                                                    self.num_feat_full)
        else:
            raise Exception("covariance_type must be 'spherical' or 'diag'")
        return distances

    def _pick_K_datapoints(self, K):
        """ Picks K datapoints at random. If the Sparsifier has access to HDX,
        it will choose from that; otherwise draws from RHDX and returns a dense
        vector (with zeros outside the mask). """
        # pick K data points at random uniformly
        rng = self.check_random_state(self.random_state)
        cluster_indices = rng.choice(self.num_samp, K, replace = False)
        cluster_indices.sort()
        # assign the cluster_centers as dense members of HDX ...
        if self.HDX is not None:
            cluster_centers_ = np.array(self.HDX[cluster_indices])
        # or assign just the M entries specified by the mask
        else:
            cluster_centers_ = np.zeros((self.K,self.num_feat_full))
            for k in range(K):
                cluster_centers_[k][mask[cluster_indices[k]]] = \
                        self.RHDX[cluster_indices[k]]
        return [cluster_centers_, cluster_indices]


    def __init__(self, num_feat_full, num_feat_comp, num_feat_shared, num_samp,
                 D_indices, transform, mask, random_state = None):

        self.num_feat_full = num_feat_full
        self.num_feat_comp = num_feat_comp
        self.num_feat_shared = num_feat_shared
        self.num_samp = num_samp
        self.transform = transform
        self.random_state = random_state

        #TODO implement default arguments for several of these

        # assign mask
        if mask is None:
            self.mask = self._generate_mask()
        else:
            self.mask = mask

        # assign D_indices
        if D_indices is None:
            self.D_indices = self._generate_D_indices()
        else:
            self.D_indices = D_indices




