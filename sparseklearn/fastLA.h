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


double _l2_distance_both_masked(double *compressed_sample_1, 
                                double *compressed_sample_2, 
                                int64_t *mask_1, 
                                int64_t *mask_2,
                                int64_t num_feat_comp,
                                int64_t num_feat_full);

double _l2_distance_one_masked_one_full(double *compressed_sample, 
                                        double *full_sample, 
                                        int64_t *mask, 
                                        int64_t num_feat_comp,
                                        int64_t num_feat_full);

void sqrt_array(double *array, const int64_t num_samples_U, const int64_t num_subsamples_X);

void pwdist(double *result, int64_t num_samples_U, int64_t num_subsamples_X, 
            struct CONSTANTS *C, struct DATA *D, int power, int64_t *S, 
            double *W, double *U, double *Sigma);
