import numpy as np
import os
import numbers
from scipy import sparse
from scipy.fftpack import dct, idct
from numpy.ctypeslib import ndpointer
import ctypes as ct

from sparseklearn.fastLA import pairwise_l2_distances_with_self
from sparseklearn.fastLA import pairwise_l2_distances_with_full
from sparseklearn.fastLA import compute_weighted_first_moment_array
from sparseklearn.fastLA import compute_weighted_first_and_second_moment_array

from sparseklearn.fastLA import pairwise_mahalanobis_distances_spherical
from sparseklearn.fastLA import pairwise_mahalanobis_distances_diagonal

class Sparsifier():
    """ Sparsifier.

    Compresses data through sparsification. Permits several operations on
    sparsified data.

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
        can be set using the D_indices parameter). The direct cosine transform
        is currently the only method supported ('dct').

    mask : np.ndarray, shape (n_datapoints, dim_mask), optional
        defaults to None. The user-provided mask. If None, mask is
        generated using the generate_mask method.

    num_feat_shared : int, defaults to 0.
        The minimum number of dimensions to be shared across all samples in the
        compressed data.

    D_indices : np.ndarray, shape (n_datapoints,), optional
        defaults to None. The user-provided diagonal of the preconditioning matrix D.
        If None, generated using the generate_D_indices method.

    Attributes
    ----------
    mask : np.ndarray, shape (num_samp, num_feat_comp)
        The mask used to sparsify the data. Array of integers, each row is the
        indices specifying which entries that sample were kept.

    D_indices : np.ndarray, shape (n_signflips,)
        Defines the preconditioning matrix D. Array of integers,
        the indices of the preconditioning matrix D with sign -1.

    """



    ###########################################################################
    # Preconditoning and mask generation

    def _generate_D_indices(self, transform):
        """ Randomly generate the D matrix in the HD transform. Store only the
        indices where D == -1. """
        rng = self.random_state
        if transform in ['dct']:
            D_indices = np.array([i for i in range(self.num_feat_full)
                if rng.choice([0,1])])
        elif transform is None:
            D_indices = None
        return D_indices

    def _generate_mask(self):
        """Generate a sparsifying mask."""
        rng = self.random_state
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

    def apply_mask(self, X, mask):
        """ Apply the mask to X.

        Parameters
        ----------

        X : np.ndarray, shape(n, P)
        mask : np.ndarray, shape(n, Q)


        Returns
        -------

        RX : np.ndarray, shape(n, Q)
            Masked X. The nth row of RX is X[n][mask[n]].

        """
        X_masked = np.array([X[n][self.mask[n]] for n in range(self.mask.shape[0])])
        return X_masked

    def apply_HD(self, X):
        """ Apply the preconditioning transform to X.

        Parameters
        ----------

        X : np.ndarray, shape (n, P)
            The data to precondition. Each row is a datapoint.


        Returns
        -------

        HDX : np.ndarray, shape (n, P)
            The transformed data.

        """
        # copy it for now
        Y = np.copy(X)
        # apply D matrix
        if self.D_indices is not None:
            Y[:,self.D_indices] *= -1
        # apply H matrix
        if self.transform == 'dct':
            Y = dct(Y, norm = 'ortho', axis = 1, overwrite_x = False)
        return Y

    def invert_HD(self, HDX):
        """ Apply the inverse of HD to HDX.

        Parameters
        ----------

        HDX : np.ndarray, shape (n, P)
            The preconditioned data. Each row is a datapoint.


        Returns
        -------

        X : np.ndarray, shape (n, P)
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

    def _set_RHDX(self, X, HDX, RHDX):
        """ Wrapper to compute RHDX from HDX or assign it if user-specified. """
        if RHDX is not None:
            return RHDX.astype(float)
        elif HDX is not None:
            return self.apply_mask(HDX.astype(float), self.mask)
        else:
            return self.apply_mask(X.astype(float), self.mask)

    def _check_X(self, X):
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("X must be a 2D array.")
        elif X.shape[0] != self.num_samp:
            raise Exception(f"X must have num_samp = {self.num_samp} rows, but has {X.shape[0]}.")
        elif X.shape[1] != self.num_feat_full:
            raise Exception(f"X must have num_feat_full = {self.num_feat_full} columns, but has {X.shape[1]}.")

    def _check_HDX(self, HDX):
        if not isinstance(HDX, np.ndarray) or HDX.ndim != 2:
            raise TypeError("HDX must be a 2D array.")
        elif HDX.shape[0] != self.num_samp:
            raise Exception(f"HDX must have num_samp = {self.num_samp} rows, but has {HDX.shape[0]}.")
        elif HDX.shape[1] != self.num_feat_full:
            raise Exception(f"HDX must have num_feat_full = {self.num_feat_full} columns, but has {HDX.shape[1]}.")

    def _check_RHDX(self, RHDX):
        if not isinstance(RHDX, np.ndarray) or RHDX.ndim != 2:
            raise TypeError("RHDX must be a 2D array.")
        elif RHDX.shape[0] != self.num_samp:
            raise Exception(f"RHDX must have num_samp = {self.num_samp} rows, but has {RHDX.shape[0]}.")
        elif RHDX.shape[1] != self.num_feat_comp:
            raise Exception(f"RHDX must have num_feat_full = {self.num_feat_comp} columns, but has {RHDX.shape[1]}.")

    def fit_sparsifier(self, X = None, HDX = None, RHDX = None,):
        """ Fit the sparsifier to specified data.

        Sets self.RHDX, the sumsampled, preconditioned data.
        At least one of the parameters must be set. If RHDX is passed,
        then X and HDX are ignored. If HDX is passed, then X is ignored.

        Parameters
        ----------

        X : np.ndarray, shape (num_samp, num_feat_full), defaults to None.
            Dense, raw data.

        HDX : np.ndarray, shape (num_samp, num_feat_full), defaults to None.
            Dense, preconditioned data.

        RHDX : np.ndarray, shape (num_samp, num_feat_comp), defaults to None.
            Subsampled, preconditioned data.
        """


        if HDX is None and RHDX is None:
            self._check_X(X)
            RHDX = self.apply_mask(self.apply_HD(X), self.mask)
        elif HDX is not None and RHDX is None:
            self._check_HDX(HDX)
            RHDX = self.apply_mask(HDX, self.mask)
        elif RHDX is not None:
            self._check_RHDX(RHDX)
        else:
            raise Exception("Must pass at least one of H, HDX, or RHDX.")

        self.RHDX = RHDX

    ###########################################################################
    # Operations on masked data

    def pairwise_distances(self, Y = None):
        """ Computes the pairwise distance between each sparsified sample,
        or between each sparsified sample and each full sample in Y if
        Y is given.

        Parameters
        ----------

        Y : np.ndarray, shape (K, P), optional
            defaults to None. Full, transformed samples.

        Returns
        -------

        distances : np.ndarray, shape(K or N, N)
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
        ----------

        W : np.ndarray, shape (N, K)
            Weights. Each row corresponds to a sample, each column to a set of
            weights. The columns of W should sum to 1. There is no necessary
            correspondence between the columns of W.

        Returns
        -------

        means : np.ndarray, shape (K,P)
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
        ----------

        W : np.ndarray, shape (N, K)
            Weights. Each row corresponds to a sample, each column to a set of
            weights. The columns of W should sum to 1. There is no necessary
            correspondence between the columns of W.

        Returns
        -------

        means : np.ndarray, shape (K,P)
            Weighted full means. Each row corresponds to a possible independent
            set of weights (for example, a binary W with K columns would give
            the means of K clusters).

        variances : np.ndarray, shape (K,P)
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

        means : np.ndarray, shape (K,P)
            The means with which to take the mahalanobis distances. Each row of
            means is a single mean in P-dimensional space.

        covariances : np.ndarray, shape (K,P) or shape (P,).
            The non-zero entries of the covariance matrix. If
            covariance_type is 'spherical', must be shape (P,). If
            covariance_type is 'diag', must be shape (K,P)

        covariance_type : {'spherical', 'diag'}, string.
            The form of the covariance matrix.

        Returns
        -------

        distances : np.ndarray, shape (N,K)
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


    def _pick_K_dense_datapoints_kmpp(self, K):
        """ Picks K datapoints randomly according to the kmpp method.

        Parameters
        ----------

        K : int
            The number of datapoints to pick.


        Returns
        -------

        datapoints : np.ndarray, shape (K,P)
            Each row is a point in dense space. If HDX is given,
            uses this. Otherwise, maps RHDX to dense space and fills in zeros.

        datapoint_indices : np.ndarray, shape (K,)
            The indices of the datapoints in self.X.
        """

        rng = self.random_state
        datapoint_indices = np.zeros(K, dtype = int)
        datapoints = np.zeros((K, self.num_feat_full))

        # pick the first one at random from the data
        datapoint_indices[0] = rng.choice(self.num_samp)
        datapoints[0][self.mask[datapoint_indices[0]]] = \
            self.RHDX[datapoint_indices[0]]

        # initialize the previous distance counter to max float
        # (so it's guaranteed to be overwritten in the loop)
        d_prev = np.ones(self.num_samp) * np.finfo(float).max

        # now pick the remaining k-1 cluster_centers
        for k in range(1,K):
            # squared distance from all the data points to the last cluster added
            latest_cluster = datapoints[k-1,np.newaxis]
            d_curr = self.pairwise_distances(Y = latest_cluster)[:,0]**2
            # ||x - U|| is either this distance or the current minimum
            where_we_have_not_improved = np.where(d_curr > d_prev)[0]
            d_curr[where_we_have_not_improved] = d_prev[where_we_have_not_improved]
            d_prev = np.copy(d_curr)

            d_curr_sum = d_curr.sum()

            # if the mask didn't obliterate all distance information, then
            # pick a datapoint at random with prob proportional to its squared
            # distance from the current cluster set
            if d_curr_sum > 0:
                datapoint_indices[k] = rng.choice(self.num_samp, p = d_curr/d_curr_sum)
            else:
                # then the mask obliterated all distance information, so just
                # pick one uniformly at random that's not already been chosen
                available_indices = set(range(self.num_samp)).difference(set(datapoint_indices))
                datapoint_indices[k] = np.random.choice(list(available_indices))
            # finally, assign the cluster, either by setting all P entires
            datapoints[k][self.mask[datapoint_indices[k]]] = \
                self.RHDX[datapoint_indices[k]]

        return [datapoints, datapoint_indices]


    def _pick_K_dense_datapoints_random(self, K):
        """ Picks K datapoints at random. If the Sparsifier has access to HDX,
        it will choose from that; otherwise draws from RHDX and returns a dense
        vector (with zeros outside the mask). """
        # pick K data points at random uniformly
        rng = self.random_state
        datapoint_indices = rng.choice(self.num_samp, K, replace = False)
        datapoint_indices.sort()
        datapoints = np.zeros((K,self.num_feat_full))
        for k in range(K):
            datapoints[k][self.mask[datapoint_indices[k]]] = \
                    self.RHDX[datapoint_indices[k]]
        return [datapoints, datapoint_indices]



    def _initialize_num_feat_full(self, num_feat_full: int):
        if isinstance(num_feat_full, numbers.Integral) and num_feat_full > 0:
            self.num_feat_full = num_feat_full
        else:
            raise Exception(f"num_feat_full must be a positive integer, but is {num_feat_full}.")

    def _initialize_num_samp(self, num_samp: int):
        if isinstance(num_samp, numbers.Integral) and num_samp > 0:
            self.num_samp = num_samp
        else:
            raise Exception(f"num_samp must be a positive integer, but is {num_samp}.")

    def _initialize_num_feat_comp(self, num_feat_comp: int, num_feat_full: int):
        if isinstance(num_feat_comp, numbers.Integral) and num_feat_comp <= num_feat_full:
            self.num_feat_comp = num_feat_comp
        else:
            raise Exception(f"num_feat_comp must be None or an integer < num_feat_full = {num_feat_full}, but is {num_feat_comp}")

    def _initialize_num_feat_shared(self, num_feat_shared: int, num_feat_full: int):
        if isinstance(num_feat_shared, numbers.Integral) and num_feat_shared <= num_feat_full:
            self.num_feat_shared = num_feat_shared
        else:
            raise Exception(f"num_feat_shared must be None or an integer < num_feat_full = {num_feat_full}, but is {num_feat_shared}")

    def _initialize_D_indices(self, D_indices: np.ndarray, num_feat_full: int, transform):
            if D_indices is None:
                self.D_indices = self._generate_D_indices(transform)
            elif isinstance(D_indices, np.ndarray):
                #TODO: check max, min, len, unique, integral
                self.D_indices = D_indices
            else:
                raise Exception(f"D_indices is type {type(num_feat_shared)}; must be array of integers or none.")

    def _initialize_random_state(self, seed):
        if seed is None or seed is np.random:
            self.random_state =  np.random.mtrand._rand
        elif isinstance(seed, (numbers.Integral, np.integer)):
            self.random_state =  np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self.random_state = seed
        else:
            raise ValueError(f"{seed} cannot be used to seed a numpy.random.RandomState")

    def _initialize_transform(self, transform):
        if transform not in [None, 'dct']:
            raise Exception(f"Transform must be one of [None, 'dct'], but is {transform}.")
        self.transform = transform

    def _initialize_mask(self, mask: np.ndarray):
        if mask is None:
            self.mask = self._generate_mask()
        elif isinstance(mask, np.ndarray):
            #TODO: check size, sorted, etc.
            self.mask = mask
        else:
            raise Exception(f"Mask must be None or array of integers; but is type {type(mask)}.")

    def __init__(self, num_feat_full, num_feat_comp, num_samp,
                 mask = None, transform = 'dct', D_indices = None, num_feat_shared = 0,
                 random_state = None):

        self._initialize_transform(transform)
        self._initialize_num_feat_full(num_feat_full)
        self._initialize_num_feat_comp(num_feat_comp, num_feat_full)
        self._initialize_num_feat_shared(num_feat_shared, num_feat_full)
        self._initialize_num_samp(num_samp)
        self._initialize_random_state(random_state)
        self._initialize_D_indices(D_indices, num_feat_full, transform)
        self._initialize_mask(mask)
