#include <stdio.h>
#include <stdint.h>
#include <math.h>

struct CONSTANTS
{
    int64_t N; // number of data points
    int64_t P; // latent dimension
    int64_t Q; // subsampled latent dimension
};

struct DATA
{
    double *RHDX;  // reduced data, N x Q
    int64_t *mask; // data mask, N x Q
};


double _l2_distance_both_compressed(double *compressed_sample_1, 
                                double *compressed_sample_2, 
                                int64_t *mask_1, 
                                int64_t *mask_2,
                                int64_t num_feat_comp,
                                int64_t num_feat_full);

double _l2_distance_one_compressed_one_full(double *compressed_sample, 
                                        double *full_sample, 
                                        int64_t *mask, 
                                        int64_t num_feat_comp,
                                        int64_t num_feat_full);

void pairwise_l2_distances_with_self(double *result,
                                     double *compressed_array,
                                     int64_t *mask_array,
                                     int64_t num_samples,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full);

void pairwise_full_distances_with_full(double *result,
                                       double *compressed_array,
                                       double *full_array,
                                       int64_t *mask_array,
                                       int64_t num_samples_comp,
                                       int64_t num_samples_full,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full);

double mahalanobis_distance_spherical(double *compressed_sample,
                                      double *full_mean,
                                      int64_t *mask,
                                      double spherical_covariance,
                                      int64_t num_feat_comp,
                                      int64_t num_feat_full);

double mahalanobis_distance_diagonal(double *compressed_sample,
                                     double *full_mean,
                                     int64_t *mask,
                                     double *diagonal_covariance,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full);

void pairwise_mahalanobis_distances_spherical(double *result,
                                             double *compressed_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *spherical_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full);

void pairwise_mahalanobis_distances_diagonal(double *result,
                                             double *compressed_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *diagonal_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full);

void update_first_moment_single_sample(double *first_moment_to_update,
                                       double *normalizer_to_update,
                                       double *compressed_sample,
                                       int64_t *mask,
                                       double weight,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full);

void update_both_moments_single_sample(double *first_moment_to_update,
                                       double *second_moment_to_update,
                                       double *normalizer_to_update,
                                       double *compressed_sample,
                                       int64_t *mask,
                                       double weight,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full);

void update_first_moment_array_single_sample(double *first_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full);

void update_both_moment_arrays_single_sample(double *first_moment_array,
                                            double *second_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full);

void sqrt_array(double *array, const int64_t num_samples_U, const int64_t num_subsamples_X);

void pwdist(double *result, int64_t num_samples_U, int64_t num_subsamples_X, 
            struct CONSTANTS *C, struct DATA *D, int power, int64_t *S, 
            double *W, double *U, double *Sigma);
