import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct
import glob

# local_path = os.path.dirname(__file__)
# dist = ct.CDLL(local_path + os.path.sep + 'source' + os.path.sep +
#                'libdistances.so')
# mom = ct.CDLL(local_path + os.path.sep + 'source' + os.path.sep +
#               'libmoments.so')
# aux = ct.CDLL(local_path + os.path.sep + 'source' + os.path.sep +
#               'libauxiliary.so')

path = os.path.abspath(__file__)
path = os.path.realpath(path)
path = os.path.dirname(path)
path = glob.glob(path + '/_fastLA*.so')[0]
_fastLA = ct.CDLL(path)

################################################################################
## dist_both_comp

_fastLA.dist_both_comp.restype = ct.c_double
_fastLA.dist_both_comp.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample_1
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample_2
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_1
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_2
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def dist_both_comp(comp_sample_1,
                            comp_sample_2,
                            mask_1,
                            mask_2,
                            num_feat_comp,
                            num_feat_full):

    return _fastLA.dist_both_comp(comp_sample_1,
                                           comp_sample_2,
                                           mask_1,
                                           mask_2,
                                           num_feat_comp,
                                           num_feat_full)

################################################################################
## dist_one_comp_one_full

_fastLA.dist_one_comp_one_full.restype = ct.c_double
_fastLA.dist_one_comp_one_full.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def dist_one_comp_one_full(comp_sample,
                                     full_sample,
                                     mask,
                                     num_feat_comp,
                                     num_feat_full):

    return _fastLA.dist_one_comp_one_full(comp_sample,
                                                   full_sample,
                                                   mask,
                                                   num_feat_comp,
                                                   num_feat_full)

################################################################################
## pairwise_l2_distances_with_self

_fastLA.pairwise_l2_distances_with_self.restype = None
_fastLA.pairwise_l2_distances_with_self.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ct.c_int64,                                   #num_samples
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_l2_distances_with_self(result,
                                    comp_array,
                                    mask_array,
                                    num_samples,
                                    num_feat_comp,
                                    num_feat_full):

    return _fastLA.pairwise_l2_distances_with_self(result,
                                                  comp_array,
                                                  mask_array,
                                                  num_samples,
                                                  num_feat_comp,
                                                  num_feat_full)


################################################################################
## pairwise_l2_distances_with_full

_fastLA.pairwise_l2_distances_with_full.restype = None
_fastLA.pairwise_l2_distances_with_full.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ct.c_int64,                                   #num_samples_comp
    ct.c_int64,                                   #num_samples_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_l2_distances_with_full(result,
                                    comp_array,
                                    full_array,
                                    mask_array,
                                    num_samples_comp,
                                    num_samples_full,
                                    num_feat_comp,
                                    num_feat_full):

    return _fastLA.pairwise_l2_distances_with_full(result,
                                                  comp_array,
                                                  full_array,
                                                  mask_array,
                                                  num_samples_comp,
                                                  num_samples_full,
                                                  num_feat_comp,
                                                  num_feat_full)

################################################################################
## mahalanobis_distance_spherical

_fastLA.mahalanobis_distance_spherical.restype = ct.c_double
_fastLA.mahalanobis_distance_spherical.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_mean
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_double,                                  #spherical_covariance
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def mahalanobis_distance_spherical(comp_sample,
                                   full_mean,
                                   mask,
                                   spherical_covariance,
                                   num_feat_comp,
                                   num_feat_full):

    return _fastLA.mahalanobis_distance_spherical(comp_sample,
                                                 full_mean,
                                                 mask,
                                                 spherical_covariance,
                                                 num_feat_comp,
                                                 num_feat_full)

################################################################################
## mahalanobis_distance_diagonal

_fastLA.mahalanobis_distance_diagonal.restype = ct.c_double
_fastLA.mahalanobis_distance_diagonal.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_mean
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'),  #diagonal covariance
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def mahalanobis_distance_diagonal(comp_sample,
                                   full_mean,
                                   mask,
                                   diagonal_covariance,
                                   num_feat_comp,
                                   num_feat_full):

    return _fastLA.mahalanobis_distance_diagonal(comp_sample,
                                                full_mean,
                                                mask,
                                                diagonal_covariance,
                                                num_feat_comp,
                                                num_feat_full)

################################################################################
## pairwise_mahalanobis_distances_spherical

_fastLA.pairwise_mahalanobis_distances_spherical.restype = None
_fastLA.pairwise_mahalanobis_distances_spherical.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_means
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #spherical_covariance_array
    ct.c_int64,                                   #num_samples_comp
    ct.c_int64,                                   #num_samples_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_mahalanobis_distances_spherical(result,
                                            comp_array,
                                            full_means,
                                            mask_array,
                                            spherical_covariance_array,
                                            num_samples_comp,
                                            num_samples_full,
                                            num_feat_comp,
                                            num_feat_full):

    return _fastLA.pairwise_mahalanobis_distances_spherical(result,
                                                          comp_array,
                                                          full_means,
                                                          mask_array,
                                                          spherical_covariance_array,
                                                          num_samples_comp,
                                                          num_samples_full,
                                                          num_feat_comp,
                                                          num_feat_full)

################################################################################
## pairwise_mahalanobis_distances_diagonal

_fastLA.pairwise_mahalanobis_distances_diagonal.restype = None
_fastLA.pairwise_mahalanobis_distances_diagonal.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #full_means
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #diagonal_covariance_array
    ct.c_int64,                                   #num_samples_comp
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def pairwise_mahalanobis_distances_diagonal(result,
                                            comp_array,
                                            full_means,
                                            mask_array,
                                            diagonal_covariance_array,
                                            num_samples_comp,
                                            num_samples_full,
                                            num_feat_comp,
                                            num_feat_full):

    return _fastLA.pairwise_mahalanobis_distances_diagonal(result,
                                                          comp_array,
                                                          full_means,
                                                          mask_array,
                                                          diagonal_covariance_array,
                                                          num_samples_comp,
                                                          num_samples_full,
                                                          num_feat_comp,
                                                          num_feat_full)


################################################################################
## update_weighted_first_moment

_fastLA.update_weighted_first_moment.restype = None
_fastLA.update_weighted_first_moment.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_to_update
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #normalizer_to_update
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_double,                                  #weight
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def update_weighted_first_moment(first_moment_to_update,
                                      normalizer_to_update,
                                      comp_sample,
                                      mask,
                                      weight,
                                      num_feat_comp,
                                      num_feat_full):

    return _fastLA.update_weighted_first_moment(first_moment_to_update,
                                                    normalizer_to_update,
                                                    comp_sample,
                                                    mask,
                                                    weight,
                                                    num_feat_comp,
                                                    num_feat_full)

################################################################################
## update_weighted_first_moment_array

_fastLA.update_weighted_first_moment_array.restype = None
_fastLA.update_weighted_first_moment_array.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #normalizer_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #weights
    ct.c_int64,                                   #num_samp_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def update_weighted_first_moment_array(first_moment_array,
                                           normalizer_array,
                                           comp_sample,
                                           mask,
                                           weights,
                                           num_samp_full,
                                           num_feat_comp,
                                           num_feat_full):

    return _fastLA.update_weighted_first_moment_array(first_moment_array,
                                                         normalizer_array,
                                                         comp_sample,
                                                         mask,
                                                         weights,
                                                         num_samp_full,
                                                         num_feat_comp,
                                                         num_feat_full)

################################################################################
## compute_weighted_first_moment_array

_fastLA.compute_weighted_first_moment_array.restype = None
_fastLA.compute_weighted_first_moment_array.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #weights
    ct.c_int64,                                   #num_samp_comp
    ct.c_int64,                                   #num_samp_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def compute_weighted_first_moment_array(first_moment_array,
                               comp_array,
                               mask_array,
                               weights_array,
                               num_samp_comp,
                               num_samp_full,
                               num_feat_comp,
                               num_feat_full):

    return _fastLA.compute_weighted_first_moment_array(first_moment_array,
                                             comp_array,
                                             mask_array,
                                             weights_array,
                                             num_samp_comp,
                                             num_samp_full,
                                             num_feat_comp,
                                             num_feat_full)

################################################################################
## update_weighted_first_and_second_moment

_fastLA.update_weighted_first_and_second_moment.restype = None
_fastLA.update_weighted_first_and_second_moment.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_to_update
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #second_moment_to_update
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #normalizer_to_update
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_double,                                  #weight
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def update_weighted_first_and_second_moment(first_moment_to_update,
                                      second_moment_to_update,
                                      normalizer_to_update,
                                      comp_sample,
                                      mask,
                                      weight,
                                      num_feat_comp,
                                      num_feat_full):

    return _fastLA.update_weighted_first_and_second_moment(first_moment_to_update,
                                                    second_moment_to_update,
                                                    normalizer_to_update,
                                                    comp_sample,
                                                    mask,
                                                    weight,
                                                    num_feat_comp,
                                                    num_feat_full)

################################################################################
## update_weighted_first_and_second_moment_array

_fastLA.update_weighted_first_and_second_moment_array.restype = None
_fastLA.update_weighted_first_and_second_moment_array.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #second_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #normalizer_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_sample
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #weights
    ct.c_int64,                                   #num_samp_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def update_weighted_first_and_second_moment_array(first_moment_array,
                                           second_moment_array,
                                           normalizer_array,
                                           comp_sample,
                                           mask,
                                           weights,
                                           num_samp_full,
                                           num_feat_comp,
                                           num_feat_full):

    return _fastLA.update_weighted_first_and_second_moment_array(first_moment_array,
                                                         second_moment_array,
                                                         normalizer_array,
                                                         comp_sample,
                                                         mask,
                                                         weights,
                                                         num_samp_full,
                                                         num_feat_comp,
                                                         num_feat_full)

################################################################################
## compute_weighted_first_and_second_moment_array

_fastLA.compute_weighted_first_and_second_moment_array.restype = None
_fastLA.compute_weighted_first_and_second_moment_array.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #first_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #second_moment_array
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #comp_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #weights
    ct.c_int64,                                   #num_samp_comp
    ct.c_int64,                                   #num_samp_full
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def compute_weighted_first_and_second_moment_array(first_moment_array,
                               second_moment_array,
                               comp_array,
                               mask_array,
                               weights_array,
                               num_samp_comp,
                               num_samp_full,
                               num_feat_comp,
                               num_feat_full):

    return _fastLA.compute_weighted_first_and_second_moment_array(first_moment_array,
                                             second_moment_array,
                                             comp_array,
                                             mask_array,
                                             weights_array,
                                             num_samp_comp,
                                             num_samp_full,
                                             num_feat_comp,
                                             num_feat_full)

################################################################################
## apply_mask_to_full_sample

_fastLA.apply_mask_to_full_sample.restype = None
_fastLA.apply_mask_to_full_sample.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #samp_full
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_int64]                                   #num_feat_comp

def apply_mask_to_full_sample(result,
                              samp_full,
                              mask,
                              num_feat_comp):

    return _fastLA.apply_mask_to_full_sample(result,
                                         samp_full,
                                         mask,
                                         num_feat_comp)

################################################################################
## logdet_cov_diag

_fastLA.logdet_cov_diag.restype = None
_fastLA.logdet_cov_diag.argtypes = [
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #result
    ndpointer(ct.c_double, flags='C_CONTIGUOUS'), #cov_array
    ndpointer(ct.c_int64, flags='C_CONTIGUOUS'),  #mask
    ct.c_int64,                                   #num_samp_comp
    ct.c_int64,                                   #num_cov
    ct.c_int64,                                   #num_feat_comp
    ct.c_int64]                                   #num_feat_full

def logdet_cov_diag(result,
                    cov_array,
                    mask,
                    num_samp_comp,
                    num_cov,
                    num_feat_comp,
                    num_feat_full):

    return _fastLA.logdet_cov_diag(result,
                               cov_array,
                               mask,
                               num_samp_comp,
                               num_cov,
                               num_feat_comp,
                               num_feat_full)
