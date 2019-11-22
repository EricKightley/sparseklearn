#include <stdio.h>
#include <stdint.h>
#include <math.h>

double dist_both_comp(double *comp_sample_1, 
                                double *comp_sample_2, 
                                int64_t *mask_1, 
                                int64_t *mask_2,
                                int64_t num_feat_comp,
                                int64_t num_feat_full);

double dist_one_comp_one_full(double *comp_sample, 
                                        double *full_sample, 
                                        int64_t *mask, 
                                        int64_t num_feat_comp,
                                        int64_t num_feat_full);

void pairwise_l2_distances_with_self(double *result,
                                     double *comp_array,
                                     int64_t *mask_array,
                                     int64_t num_samples,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full);

void pairwise_full_distances_with_full(double *result,
                                       double *comp_array,
                                       double *full_array,
                                       int64_t *mask_array,
                                       int64_t num_samples_comp,
                                       int64_t num_samples_full,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full);

double mahalanobis_distance_spherical(double *comp_sample,
                                      double *full_mean,
                                      int64_t *mask,
                                      double spherical_covariance,
                                      int64_t num_feat_comp,
                                      int64_t num_feat_full);

double mahalanobis_distance_diagonal(double *comp_sample,
                                     double *full_mean,
                                     int64_t *mask,
                                     double *diagonal_covariance,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full);

void pairwise_mahalanobis_distances_spherical(double *result,
                                             double *comp_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *spherical_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full);

void pairwise_mahalanobis_distances_diagonal(double *result,
                                             double *comp_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *diagonal_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full);
