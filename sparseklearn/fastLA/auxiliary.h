#include <stdio.h>
#include <stdint.h>
#include <math.h>

void apply_mask_to_full_sample(double *result,
                               double *samp_full, 
                               int64_t *mask, 
                               int64_t num_feat_comp);

void logdet_cov_diag(double *result,
                     double *diagonal_covariance_array,
                     int64_t *mask,
                     int64_t num_samp_comp,
                     int64_t num_cov,
                     int64_t num_feat_comp,
                     int64_t num_feat_full);
