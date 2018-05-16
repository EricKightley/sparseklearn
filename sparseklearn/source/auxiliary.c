#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "moments.h"

void apply_mask_to_full_sample(double *result,
                               double *samp_full, 
                               int64_t *mask, 
                               int64_t num_feat_comp)
/* Apply a mask to a full sample. 
 *
 * Parameters
 * ----------
 *
 * samp_full : array, length num_feat_full. The full sample to be masked.
 *
 * mask_1 : array, length num_feat_comp. The indices specifying which 
 *          entries of samp_full to keep. Must be sorted.
 *
 * num_feat_comp : The number of features in a compressed sample.
 *
 * Returns
 * -------
 *
 * result : (modified) array, length num_feat_comp. The compressed sample.
 */
{
    int64_t ind_feat_comp;
    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp ++) {
        result[ind_feat_comp] = samp_full[mask[ind_feat_comp]];
    }
}

void logdet_cov_diag(double *result,
                     double *diagonal_covariance_array,
                     int64_t *mask_array,
                     int64_t num_samp_comp,
                     int64_t num_cov,
                     int64_t num_feat_comp,
                     int64_t num_feat_full)
/* Compute the log of the determinant of the masked diagonal covariance matrix.
 *
 * Parameters
 * ----------
 *
 * diagonal_covariance_array : array, size num_samp_full by num_feat_full. 
 *                             Each row is the diagonal covariance for the 
 *                             corresponding mean.              
 *
 * mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *              indicating which entries were kept of the full datapoint from 
 *              which the comp sample in comp_array was obtained.
 *
 * num_samp_comp : the number of samples (rows) in comp_array
 *
 * num_cov: the number of components (number of entries in diagonal_covariance_array)
 *
 * num_feat_comp : the number of dimensions in the comp data
 *
 * num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 * result : array, size num_samp_full by num_cov. The i,j entry is the logdet
 *          of the jth component using the ith mask. 
 */
{
    int64_t ind_samp_comp; //indexes rows of mask_array
    int64_t ind_feat_comp; //indexes cols of mask_array
    int64_t ind_cov; //indexes rows of diagonal_covariance_array
    int64_t ind_result_entry; //indexes entry of result (row*ncol + col)
    int64_t ind_cov_entry; //indexes entry of cov (row*ncol + col)
    int64_t ind_mask_entry; //indexes entry of mask_array (row*ncol + col)

    for (ind_samp_comp = 0 ; ind_samp_comp < num_samp_comp ; ind_samp_comp ++) {
        for (ind_cov = 0 ; ind_cov < num_cov ; ind_cov ++) {
            // index of this entry of result
            ind_result_entry = ind_samp_comp * num_cov + ind_cov;
            // update result with all covariances preserved by this mask row
            for (ind_feat_comp = 0 ; ind_feat_comp < num_cov ; ind_feat_comp ++) {
                ind_mask_entry = ind_samp_comp * num_feat_comp + ind_feat_comp;
                ind_cov_entry = ind_cov * num_feat_full + mask_array[ind_mask_entry];
                result[ind_result_entry] += log(diagonal_covariance_array[ind_cov_entry]);
            }
        }
    }
}

