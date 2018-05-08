#include <stdio.h>
#include <stdint.h>
#include <math.h>

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

void compute_first_moment_array(double *first_moment_array,
                                double *compressed_array,
                                int64_t *mask_array,
                                double *weights_array,
                                int64_t num_samp_comp,
                                int64_t num_samp_full,
                                int64_t num_feat_comp,
                                int64_t num_feat_full);

void compute_both_moment_arrays(double *first_moment_array,
                                double *second_moment_array,
                                double *compressed_array,
                                int64_t *mask_array,
                                double *weights_array,
                                int64_t num_samp_comp,
                                int64_t num_samp_full,
                                int64_t num_feat_comp,
                                int64_t num_feat_full);
