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

void sqrt_array(double *array, const int64_t nrow, const int64_t ncol);

void pwdist(double *result, int64_t nrow, int64_t ncol, struct CONSTANTS *C, struct DATA *D,
            int power, int64_t *S, double *W, double *U, double *Sigma);
