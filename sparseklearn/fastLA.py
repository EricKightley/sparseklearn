import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct

fastLA = ct.CDLL('/home/eric/Dropbox/EricStephenShare/sparseklearn/sparseklearn/libfastLA.so')

################################################################################
## _l2_distance_both_masked

fastLA._l2_distance_both_masked.restype = ct.c_double
fastLA._l2_distance_both_masked.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_sample_1
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_sample_2
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_1
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_2
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def _l2_distance_both_masked(compressed_sample_1, 
                            compressed_sample_2, 
                            mask_1,
                            mask_2,
                            num_feat_comp,
                            num_feat_full):
    """ Computes the l2 distance between compressed_sample_1 and 
    compressed_sample_2, by finding the intersection of their masks, computing
    the distance between the samples projected onto this common subspace, and
    then scaling this distance back up.

    Inputs
    ------

        compressed_sample_1 : array, length num_feat_comp

        compressed_sample_2 : array, length num_feat_comp

        mask_1 : array, length num_feat_comp. The indices specifying which 
                 entries of the dense sample_1 were kept. Must be sorted.

        mask_2 : array, length num_feat_comp. The indices specifying which 
                 entries of the dense sample_2 were kept. Must be sorted.

        num_feat_comp : the number of features in a compressed sample. 

        num_feat_full : the number of features in a full sample. 

    Returns
    -------

         distance : double, the approximate l2 distance between both samples.
    """

    return fastLA._l2_distance_both_masked(compressed_sample_1,
                                           compressed_sample_2,
                                           mask_1,
                                           mask_2,
                                           num_feat_comp,
                                           num_feat_full)


"""
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
    ct.c_int64,                                   # num_samples_U
    ct.c_int64,                                   # num_subsamples_X
    ct.POINTER(CONSTANTS),                        # struct CONSTANTS
    ct.POINTER(DATA),                             # struct DATA
    ct.c_int,                                     # power
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  # S
    DoubleArrayType,                              # W
    DoubleArrayType,                              # U
    DoubleArrayType]                              # Sigma

def pairwise_distances(RHDX, mask, P, S = None, W = None, U = None, Sigma = None, power = 1):

    if (U is not None) and (U.ndim != 2):
        raise Exception('U must be a 2D array or None.')

    # assign Q
    N,Q = RHDX.shape

    # assign S if needed
    if S is None:
        S = np.array(range(N))

    num_samples_U = len(S)

    # assign num_samples_U
    if U is not None:
        num_subsamples_X = U.shape[0]
    else:
        num_subsamples_X = num_samples_U
  

    # initialize result
    result = np.zeros((num_samples_U, num_subsamples_X))
    
    # build the structs
    C = CONSTANTS(N,P,Q)
    D = DATA(RHDX.ctypes.data_as(ct.POINTER(ct.c_double)),
             mask.ctypes.data_as(ct.POINTER(ct.c_int64)))

    # call C function, which modifies result
    pwdist(result, num_samples_U, num_subsamples_X, C, D, power, S, W, U, Sigma)

    return result

"""
