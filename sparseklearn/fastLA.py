import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct

# load the shared library and assign its functions
fastLA = ct.CDLL('/home/eric/Dropbox/EricStephenShare/sparseklearn/sparseklearn/libfastLA.so')
polycomb = fastLA.polycomb
pwdist = fastLA.pwdist

# below function from
# https://stackoverflow.com/questions/64120178/how-can-i-pass-null-to-an-external-library-using-ctypes-with-an-argument-decla
def wrapped_ndptr(*args, **kwargs):
    base = ndpointer(*args, **kwargs)
    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)
    return type(base.__name__, (base,), {'from_param': classmethod(from_param)})
DoubleArrayType = wrapped_ndptr(dtype=np.float64, flags='C_CONTIGUOUS')

class CONSTANTS(ct.Structure):
    _fields_ = [
        ('N', ct.c_int64),
        ('P', ct.c_int64),
        ('Q', ct.c_int64)]

class DATA(ct.Structure):
    _fields_ = [
        ('RHDX', ct.POINTER(ct.c_double)),
        ('mask', ct.POINTER(ct.c_int64))]

polycomb.restype = None
# make the arguments that should be optional (W, U, Sigma)
# use the DoubleArrayType, the other ndarrays will use ndpointer
polycomb.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'),
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'),
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),
    DoubleArrayType,
    DoubleArrayType,
    DoubleArrayType,
    ct.c_int64,
    ct.c_int64,
    ct.c_int64,
    ct.c_int64,
    ct.c_int64,
    ct.c_int64
]

def polynomial_combination(X, mask, P, S = None, W = None, U = None, Sigma = None, power = 1):
    """

    Compute sum_n w_nk (x_n - R_n U_k)^power // Sigma_k

    Inputs
    ------

    X : array of floats, shape (N,Q)
        The sparsified data. Each row is a sparsified datapoint.

    mask : array of integers, shape (N,Q)
        The sparsifying mask. Each row specifies which entries of the 
        corresponding dense datapoint have been kept. 

    P : int
        The latent dimension (the length of a raw datapoint X, 
        before subsampling).

    S : array of ints, optional
        Subset of the data X to use. Each entry in S corresponds
        to a row of X. If None, all rows of X are used.

    W : array of floats, shape (N,K), optional
        Data weights. If None, K is inferred from mu and W is taken to be all 1s.

    U : array of floats, shape (K,P), optional
        Offsets. If None, K is taken to be 1 and U is taken to be all 0s.

    Sigma : array of floats, shape (K,P), optional
        Dimension weights. If None, K is inferred from U and Sigma is taken to
        be all 1s.

    power : int, optional
        Polynomial exponent, defaults to 1. 


    Returns
    -------

    result : array of floats, shape (K, P)
        The sum above.

    """    
    # assign Q
    N,Q = X.shape

    # assign K
    if U is not None:
        K = U.shape[0]
    elif W is not None:
        K = W.shape[1]
    elif Sigma is not None:
        K = Sigma.shape[0]
    else:
        K = 1

    # assign S if we need to
    if S is None:
        S = np.array(range(N))

    # initialize result
    result = np.zeros((K,P))
    # call C function, which modifies result
    polycomb(result, X, mask, S, W, U, Sigma, power, len(S), Q, N, P, K)

    return result

pwdist.restype = None
pwdist.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), # result
    ct.c_int64,                                   # nrow
    ct.c_int64,                                   # ncol
    ct.POINTER(CONSTANTS),                        # struct CONSTANTS
    ct.POINTER(DATA),                             # struct DATA
    ct.c_int,                                     # power
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  # S
    DoubleArrayType,                              # W
    DoubleArrayType,                              # U
    DoubleArrayType]                              # Sigma

def pairwise_distances(RHDX, mask, S, W, U, Sigma, power, P):
    if (U is not None) and (U.ndim != 2):
        raise Exception('U must be a 2D array or None.')

    # assign Q
    N,Q = RHDX.shape

    # assign S if needed
    if S is None:
        S = np.array(range(N))

    # assign nrow
    if U is not None:
        nrow = U.shape[0]
    else:
        nrow = len(S)
  
    # assign output columns
    ncol = len(S)

    # initialize result
    result = np.zeros((nrow, ncol))
    
    # build the structs
    C = CONSTANTS(N,P,Q)
    D = DATA(RHDX.ctypes.data_as(ct.POINTER(ct.c_double)),
             mask.ctypes.data_as(ct.POINTER(ct.c_int64)))


    # call C function, which modifies result
    pwdist(result, nrow, ncol, C, D, power, S, W, U, Sigma)

    return result

