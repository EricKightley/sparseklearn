import numpy as np
import h5py
from scipy import sparse
from scipy.fftpack import dct, idct

class Sparsifier():
    """ Sparsifier.

    Compresses data through sparsification. Permits several operations on
    sparisified data. 
    
    Parameters
    ----------

    compression_target : float or int, defaults to 1.0.
        The target compression factor, indicating either the target ratio of 
        dimensions to keep (if a float in (0,1]) or the number of dimensions
        to keep (if an integer). Note that passing integer 1 will keep 1 
        dimension, whereas passing float 1.0 will keep all dimensions. 

    alpha_target : float or int, defaults to 0.0.
        The shared compression factor target, indicating either the target
        ratio of preserved dimensions to be common across all data points
        (if a float in (0,1]), or the number of dimensions to be shared
        (if an integer). 

    transform : {'dct', None}, defaults to 'dct'.
        The preconditioning transform.
        Determines what form of H to use in the preconditioning transform HD. 
        Any method other than None will also use the diagonal D matrix (which 
        can be set using the precond_D parameter). Must be one of::
            
            'dct'  discrete cosine transform
            None no transform

    mask : nd.array, shape (n_datapoints, dim_mask), optional
        defaults to None. The user-provided mask. If None, mask is
        generated using the generate_mask method.

    precond_D : nd.array, shape (n_datapoints,), optional
        defaults to None. The user-provided diagonal of the preconditioning matrix D. 
        If None, generated using the generate_D method. 

    data_dim : int, required and allowed if and only if transform is 'R' or 'RHD'
        defaults to None. 
        Dimension of each data point in the original, dense space. Will be taken
        to be the number of columns of the data if the dense data is passed in.

    Attributes
    ----------

    mask : nd.array, shape (n_datapoints, dim_mask)
        The mask used to sparsify the data. Array of integers, each row is the
        indices specifying which entries of X[row] were kept. 

    D_indices : nd.array, shape (n_signflips,)
        Defines the preconditioning matrix D. Array of integers, 
        the indices of the preconditioning matrix D with sign -1.

    compression : float
        The compression factor. Computed as Q/P where Q as been chosen to 
        make Q/P as close to compression_target as possible.

    alpha : float
        The shared compression factor. Computed as Qs/Q where Qs has been
        chosen to make Qs/Q as close to compression_target as possible

    Q : int
        Dimensionality of the mask, i.e., the number of dimensions preserved
        in each datapoint.

    Qs : int
        The number of dimensions in the mask guaranteed to be shared by all
        datapoints.

    """

    ###########################################################################
    # Input Checks

    def _input_checker_sparsifier(self, compression_target, alpha_target, transform, 
            mask, precond_D, data_dim):
        """ Check the input parameters for errors or contradictions. Only performs checks
        that can be run at instantiation. More will happen during fitting. 
        """

        # Check type and bounds on compression_target
        if (type(compression_target) is float and compression_target <= 0 or compression_target > 1) or \
            (type(compression_target) is int and compression_target <= 0) or \
            (type(compression_target) not in [int, float]):
                raise ValueError('compression_target = {}, '.format(compression_target) + 
                    'but must be a positive integer or a float 0 < c <= 1')

        # Check type and bounds on alpha_target
        if (type(alpha_target) is float and alpha_target < 0 or alpha_target > 1) or \
            (type(alpha_target) is int and (alpha_target <= 0)) or \
            (type(alpha_target) not in [int, float]):
                raise ValueError('alpha_target = {}, '.format(alpha_target) +
                    'but must be a positive integer or a float 0 <= c <= 1')

        # Check that alpha_target <= compression_target if both are ints
        if (type(compression_target) is int and type(alpha_target) is int) \
            and alpha_target > compression_target:
            raise ValueError(
                'alpha_target = {} > {} = compression_target'.format(alpha_target, compression_target))

        # Make sure precond_D is a 1-D array of integers
        if (precond_D is not None):
            if (type(precond_D) is not np.ndarray) or (precond_D.ndim !=1) or \
                (not np.issubdtype(precond_D.dtype, np.integer)):
                raise ValueError('precond_D must be an array of integers.')

        # Make sure transform is valid
        if transform not in ['dct', None]:
            raise ValueError('Transform parameter {} is not one of the '.format(transform) +
                    'available methods. See documentation.')

        # Make sure mask is a 2-D array of integers if it's given
        if (mask is not None):
            if (type(mask) is not np.ndarray) or mask.ndim != 2 or \
                (not np.issubdtype(mask.dtype, np.integer)):
                raise ValueError('Mask must be a 2D array of 64-bit integers.')


    def _data_checker(self, X, HDX, RHDX, mask, data_dim, precond_D):
        """ Perform checks on input data to make sure correct combinations
        were passed. Dimension checks occur in _dimension_checker. 
        """

        # if RHDX was passed, mask must also be passed
        if RHDX is not None and mask is None:
            raise ValueError('Subsampled data RHDX passed as input, this ' + 
                    'requires that mask also be passed.')
        # if RHDX was passed we need a way to find the ambient dimension.
        # we can do this using X or HDX if they were passed, but if they
        # weren't then we need data_dim.
        if RHDX is not None and HDX is None and X is None and data_dim is None:
            raise ValueError('Subsampled data RHDX passed as input without ' +
                    'X or HDX, so data_dim must be specified.')

        # if HDX was passed we also need precond_D to have been passed
        if HDX is not None and precond_D is None:
            raise ValueError('Preconditioned data HDX passed as input without ' +
                    'precond_D.')
        #TODO
        ## X must be a 2-D numpy array
        #if (type(X) is not np.ndarray) or (X.ndim != 2):
        #    raise ValueError('X must be a 2-D array.')
        # If the data is already subsampled, we need to know the mask
        #if transform in ['R', 'RHD'] and mask is None:
        #    raise ValueError('Transform parameter {} indicates'.format(transform) +
        #            'that the input is subsampled but mask is not given.')
        # Make sure data_dim is given if we need it and not if we don't
        #if ((transform in ['R', 'RHD']) and (type(data_dim) is not int)) or \
        #        (transform not in ['R', 'RHD'] and data_dim is not None):
        #    raise ValueError('P must be given if and only if X is already subsampled.')
        # If the data is already preconditioned, we need to know the D matrix
        #if transform == 'dct' and precond_D is None:
        #    raise ValueError('Transform parameter {} indicates '.format(transform) +
        #            'that the input is preconditioned but precond_D is not given.')
        #
        # RHDX and compression_target can't both be given
        return

    def _dimension_checker(self, compression, alpha, Q, Qs, N, P, X, HDX, RHDX):
        #TODO
        # check that data_dim and HDX and X shape agree if they are both passed
        # require data_dim if RHDX is passed but neither HDX nor X were
        return

    ###########################################################################
    # Preconditoning and mask generation

    def _generate_D(self, transform, P):
        """ Randomly generate the D matrix in the HD transform. Store only the 
        indices where D == -1.
        """
        # ... generate it randomly if we are using it ...
        if transform in ['dct']:
            D_indices = np.array([i for i in range(P) 
                if np.random.choice([0,1])])
        # ... or set it empty if we are not using it.
        else:
            D_indices = np.array([])
        return D_indices

    def _set_D(self, transform, precond_D, P):
        """ Wrapper for _generate_D, which will take the user-specified
        precond_D if given. 
        """
        if precond_D is None:
            D_indices = self._generate_D(transform, P)
        else:
            D_indices = precond_D
        return D_indices

    def _generate_mask(self, P, Qs, Qr, N):
        """ Randomly generate the sparsifying mask.
        """
        inds = [p for p in range(P)]
        np.random.shuffle(inds)
        shared_mask = inds[:Qs]
        inds = inds[Qs:]
        # if there are any random indices to generate, do so
        if Qr > 0:
            # generate the random masks
            random_masks = [np.random.choice(inds, Qr, replace=False)
                            for n in range(N)]
            # concatenate them with the shared masks
            mask = np.concatenate((random_masks, 
                   np.tile(shared_mask, (N,1)).astype(int)), axis = 1)
        else:
            # then there are no random ones to generate
            mask = np.tile(shared_mask, (N,1)).astype(int)

        mask.sort(axis=1)
        return mask

    def _set_mask(self, mask, P, Qs, Qr, N):
        """ Wrapper for _generate_mask, which will take the user-specified mask
        instead if given.
        """
        if mask is not None:
            return mask
        else:
            return self._generate_mask(P, Qs, Qr, N)


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
        col_inds = [n for n in range(self.N) for m in range(self.M)]
        row_inds = list(self.mask.flatten())
        data = np.ones_like(row_inds)
        mask_binary = sparse.csr_matrix( (data, (row_inds, col_inds)), 
                      shape = (self.P,self.N), dtype = bool)
        return mask_binary


    ###########################################################################
    # Fitting

    def _compute_constants(self, X, HDX, RHDX, compression_target, alpha_target, data_dim):
        """ Compute the constants needed for the sparsifier. As usual, at least
        one of X, HDX, or RHDX must not be None. 

        Parameters
        ----------

        X : nd.array, (N, P)
            Dense, raw data. Can be None.

        HDX : nd.array, (N,P)
            Dense, transformed data. Can be None.

        RHDX : nd.array, (N,Q)
            Subsampled, transformed data. Can be None. 

        compression_target : see _init_

        alpha_target : see _init_

        Returns
        -------


        compression_adjusted : float
            The compression factor. Computed as Q/P where Q as been chosen to 
            make Q/P as close to compression_target as possible.

        alpha_adjusted : float
            The shared compression factor. Computed as Qs/Q where Qs has been
            chosen to make Qs/Q as close to compression_target as possible

        Q : int
            Dimensionality of the mask, i.e., the number of dimensions preserved
            in each datapoint.

        Qs : int
            The number of dimensions in the mask guaranteed to be shared by all
            datapoints.

        Qr : int
            The number of dimensions in the mask chosen at random. Some may
            still be shared by all datapoints by chance. 

        N : int
            The number of datapoints, computed as the number of rows in the
            input X, HDX, or RHDX.

        P : int
            The dimension of each dense datapoint (aka the ambient or latent 
            dimension). Taken to be the number of columns of X or HDX, or
            required as input if only RHDX is given. 
        """
        # if RHDX is given, use this to compute the compression factor.
        if RHDX is not None:
            # RHDX is reduced, so this sets Q
            N,Q = np.shape(RHDX)
            # Get P from HDX if we can:
            if HDX is not None:
                P = HDX.shape[1]
            # ... or from X
            elif X is not None:
                P = X.shape[1]
            # ... or finally from user-specified P if we need to
            else:
                P = data_dim
            # We don't need shared/random split for the mask.
            Qs = None
            Qr = None
            compression_adjusted = Q/P
            alpha_adjusted = None

        # if RHDX was not given, we need to compute the subsampling constants
        else:
            # if we have X use its shape
            if X is not None:
                N,P = np.shape(X)
            # otherwise use HDX's shape
            elif HDX is not None:
                N,P = np.shape(HDX)
            # if compression_target is a float, find the number of dimensions:
            if type(compression_target) is float:
                Q = int(np.floor(P * compression_target))
            # if compression_target is given as the num dim, find the ratio
            if (type(compression_target) is int) or \
                (type(compression_target) is np.int64):
                Q = compression_target
            # compute number of shared and random subsampling
            if type(alpha_target) is float:
                # then alpha is the ratio of shared indices
                Qs = int(np.floor(Q * alpha_target))
            elif type(alpha_target) is int or type(alpha_target) is np.int64:
                # then alpha is the number of shared indices
                Qs = alpha_target
            # if we are not compressing then set Qs to Q
            if compression_target == 1.0:
                Qs = Q
            Qr = Q - Qs
            # recompute alpha and compression (either because they were ints or to 
            # account for rounding from floor function above)
            compression_adjusted = Q/P
            alpha_adjusted = Qs/Q
        return([compression_adjusted, alpha_adjusted, Q, Qs, Qr, N, P])
    
    def _set_HDX(self, transform, X, HDX, RHDX):
        """ Wrapper to compute HDX from X or assign it if user-specified. """
        if HDX is not None:
            return HDX
        elif (X is not None and transform == 'dct'):
            return self.apply_HD(X)
        else:
            return None

    def _set_RHDX(self, X, HDX, RHDX):
        """ Wrapper to compute RHDX from HDX or assign it if user-specified. """
        if RHDX is not None:
            return RHDX
        elif HDX is not None:
            return self.apply_mask(HDX)
        else:
            return self.apply_mask(X)

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

        # check input data
        self._data_checker(X, HDX, RHDX, self.mask, self.data_dim, self.precond_D)
        # set model constants 
        self.compression, self.alpha, self.Q, self.Qs, self.Qr, self.N, self.P = \
                self._compute_constants(X, HDX, RHDX, self.compression_target, self.alpha_target,
                        self.data_dim)
        # run checks
        self._dimension_checker(self.compression, self.alpha, self.Q, 
                self.Qs, self.N, self.P, X, HDX, RHDX)
        # set D_indices
        self.D_indices = self._set_D(self.transform, self.precond_D, self.P)
        # set mask
        self.mask = self._set_mask(self.mask, self.P, self.Qs, self.Qr, self.N) 
        # compute HDX and RHDX
        HDX = self._set_HDX(self.transform, X, HDX, RHDX)
        RHDX = self._set_RHDX(X, HDX, RHDX)
        # assign data
        self.X = X
        self.HDX = HDX
        self.RHDX = RHDX



    ###########################################################################
    # Operations on masked data

    def polynomial_combination(self, W, power = 1):
        """

        Computes a weighted sum over the masked data, with powers.

        .. math::

           \sum_{n} W_{nk} R_n^T R_n{\mathbf{x}_n}^p

        Parameters
        ----------

            W : nd.array, shape (N,k)
                Weights to apply to the terms in the sum.
            power : int or float, optional
                The exponent, defaults to 1. 

        Returns
        -------

            comb : nd.array, shape (k,P)
                The weighted sum over the exponentiated data.

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
        """ Compute the pairwise distances between the masked X and Y.

        Parameters
        ----------

        Y : nd.array, shape (k,P) or (k,Q), optional
            The array of vectors, each row is a point, default = None. If None,
            is taken to be X. 
        W : nd.array, shape (k,P) or (k,Q), optional
            The weights for each point, default = None. If None, is taken to
            be vector of ones.
        transform_Y : str, temporary
        transform_W : str, temporary

        Returns
        -------

        dist : nd.array, shape (N,k)
            The distances. dist[n,j] is the distance of the nth datapoint to Y[j].

        """


        X = self.RHDX

        # transform Y if we need to
        if "HD" in transform_Y and Y is not None:
            Y = self.apply_HD(Y)
        # assign Y to be X if we need to
        elif Y is None:
            Y = X
        # augment Y if we need to
        if Y.ndim == 1:
            Y = Y[np.newaxis,:]
        
        # transform W if we need to
        if "HD" in transform_W and W is not None:
            W = self.apply_HD(W)

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
                    y = self.apply_mask(Y[k])
                else:
                    y = Y[k]
                dist[k] = np.linalg.norm(y - X, axis = 1)
        # ... or alternatively if they are given
        else:
            for k in range(K):
                if subsample_Y:
                    y = self.apply_mask(Y[k])
                else:
                    y = Y[k]
                if subsample_W:
                    w = self.apply_mask(W[k])
                else:
                    w = W[k]
                Xmys = (X-y)**2
                dist[k] = np.sqrt([np.dot(Xmys[n], w[n]) for n in range(self.N)])
        return dist.T

    def _pick_K_datapoints(self, K):
        """ Picks K datapoints at random. If the Sparsifier has access to HDX,
        it will choose from that; otherwise draws from RHDX and returns a dense
        vector (with zeros outside the mask). """
        # pick K data points at random uniformly
        cluster_indices = np.random.choice(self.N, K, replace = False)
        cluster_indices.sort()
        # assign the cluster_centers as dense members of HDX ...
        if self.HDX is not None:
            cluster_centers_ = np.array(self.HDX[cluster_indices])
        # or assign just the M entries specified by the mask
        else:
            cluster_centers_ = np.zeros((self.K,self.P))
            for k in range(K):
                cluster_centers_[k][mask[cluster_indices[k]]] = \
                        self.RHDX[cluster_indices[k]]
        return [cluster_centers_, cluster_indices]


    def __init__(self, compression_target = 1.0, alpha_target = 0.0, transform = 'dct',
            mask = None, precond_D = None, data_dim = None):

        self._input_checker_sparsifier(compression_target, alpha_target, transform, mask, 
                precond_D, data_dim)

        self.compression_target = compression_target
        self.alpha_target = alpha_target
        self.transform = transform
        self.mask = mask
        self.precond_D = precond_D
        self.data_dim = data_dim

