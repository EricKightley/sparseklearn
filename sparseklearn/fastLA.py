import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct

fastLA = ct.CDLL('/home/eric/Dropbox/EricStephenShare/sparseklearn/sparseklearn/libfastLA.so')

################################################################################
## _l2_distance_both_compressed

fastLA._l2_distance_both_compressed.restype = ct.c_double
fastLA._l2_distance_both_compressed.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_sample_1
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_sample_2
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_1
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_2
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def _l2_distance_both_compressed(compressed_sample_1, 
                            compressed_sample_2, 
                            mask_1,
                            mask_2,
                            num_feat_comp,
                            num_feat_full):

    return fastLA._l2_distance_both_compressed(compressed_sample_1,
                                           compressed_sample_2,
                                           mask_1,
                                           mask_2,
                                           num_feat_comp,
                                           num_feat_full)

################################################################################
## _l2_distance_one_compressed_one_full

fastLA._l2_distance_one_compressed_one_full.restype = ct.c_double
fastLA._l2_distance_one_compressed_one_full.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_sample
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def _l2_distance_one_compressed_one_full(compressed_sample, 
                                     full_sample, 
                                     mask,
                                     num_feat_comp,
                                     num_feat_full):

    return fastLA._l2_distance_one_compressed_one_full(compressed_sample,
                                                   full_sample,
                                                   mask,
                                                   num_feat_comp,
                                                   num_feat_full)

################################################################################
## pairwise_l2_distances_with_self

fastLA.pairwise_l2_distances_with_self.restype = None
fastLA.pairwise_l2_distances_with_self.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ct.c_int64,                                   #num_samples
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_l2_distances_with_self(result,
                                    compressed_array,
                                    mask_array,
                                    num_samples,
                                    num_feat_comp,
                                    num_feat_full):

    return fastLA.pairwise_l2_distances_with_self(result,
                                                  compressed_array,
                                                  mask_array,
                                                  num_samples,
                                                  num_feat_comp,
                                                  num_feat_full)


################################################################################
## pairwise_l2_distances_with_full

fastLA.pairwise_l2_distances_with_full.restype = None
fastLA.pairwise_l2_distances_with_full.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #compressed_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ct.c_int64,                                   #num_samples_comp
    ct.c_int64,                                   #num_samples_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_l2_distances_with_full(result,
                                    compressed_array,
                                    full_array,
                                    mask_array,
                                    num_samples_comp,
                                    num_samples_full,
                                    num_feat_comp,
                                    num_feat_full):

    return fastLA.pairwise_l2_distances_with_full(result,
                                                  compressed_array,
                                                  full_array,
                                                  mask_array,
                                                  num_samples_comp,
                                                  num_samples_full,
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
