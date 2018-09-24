#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "moments.h"

double GREATER_THAN_ZERO_TOL = 1E-16; 

///////////////////////////////////////////////////////////////////////////////
//  First and Second Moments

void update_weighted_first_moment(double *first_moment_to_update,
                                       double *normalizer_to_update,
                                       double *compressed_sample,
                                       int64_t *mask,
                                       double weight,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full)
/* Performs an update to the first moment using a single compressed_sample
 * and weight. Also updates the normalizer. The normalizer is needed because
 * the weights must sum to 1 over all samples, but due to sparsification, not
 * all weights will be used on all dimensions of the full mean, so each entry
 * in the num_feat_full-dimensional space must be individually renormalized by
 * the sum of all weights that were used to modify that specific entry.
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     weight : the weight associated with compressed_sample
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 *
 * Returns
 * -------
 *
 *      first_moment_to_update : (modified) array, len num_feat_full. 
 *                               The current first moment (mean), to be 
 *                               updated.
 *
 *      second_moment_to_update : (modified) array, len num_feat_full. 
 *                               The current first moment (mean), to be 
 *                               updated.
 *
 *      normalizer_to_update : (modified) array, len num_feat_full. The 
 *                             current normalizer, used to keep track of
 *                             which weights have been used in which 
 *                             dimensions.
 */
{ 
    int64_t ind_feat_comp = 0; //indexes through compressed_sample and mask
    int64_t ind_feat_full = 0; //indexes through moment and normalizer
    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp++ ) {
        ind_feat_full = mask[ind_feat_comp];
        first_moment_to_update[ind_feat_full] += weight * compressed_sample[ind_feat_comp];
        normalizer_to_update[ind_feat_full] += weight;
    }
}

void update_weighted_first_moment_array(double *first_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full)
/* Update a set of first moments using a single sample. Wrapper for
 * update_weighted_first_moment; see that function's docs for details.
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     weights : array, length num_samp_full, ith entry is the weight
 *               associated with compressed_sample for the ith moment. 
 *
 *     num_samp_full : the number of moments to be updated. 
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      first_moment_array : (modified) array, size num_samp_full by 
 *                           num_feat_full. Each row is a first moment (mean)
 *                           to be updated.
 *
 *      normalizer_array : (modified) array, size num_samp_full by num_feat_full. 
 *                         Each row is a normalizer, used to keep track of
 *                         which weights have been used in which dimensions.
 * */
{ 
    int64_t ind_samp_full = 0; //indexes rows of first_moment_array
    for (ind_samp_full = 0 ; ind_samp_full < num_samp_full ; ind_samp_full ++) {
        update_weighted_first_moment(&first_moment_array[ind_samp_full*num_feat_full],
                                          &normalizer_array[ind_samp_full*num_feat_full],
                                          compressed_sample,
                                          mask,
                                          weights[ind_samp_full],
                                          num_feat_comp,
                                          num_feat_full);
    }
}

void compute_weighted_first_moment_array(double *first_moment_array,
                                double *compressed_array,
                                int64_t *mask_array,
                                double *weights_array,
                                int64_t num_samp_comp,
                                int64_t num_samp_full,
                                int64_t num_feat_comp,
                                int64_t num_feat_full)
/* Compute weighted first moments. Every column of weights_array (and
 * correspondingly first_moment_array) corresponds to a set of weights.
 * For each set of weights, compute the sum of the samples in compressed_array
 * (the rows), weighted by the elements of this set of weights. 
 *
 * Wrapper for update_weighted_first_moment_array, calling this function
 * for each row of compressed_array. 
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samp_comp by num_feat_comp. Each row 
 *                        is a datapoint in the compressed domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the compressed sample in compressed_array was obtained.
 *
 *     weights_array : array, size num_samp_comp by num_samp_full.
 *                     ith col is the weights associated with each row of
 *                     compressed_sample for the ith moment. 
 *
 *     num_samp_comp : the number of samples (rows) in compressed_array
 *
 *     num_samp_full : the number of moments to be updated. 
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      first_moment_array : (modified) array, size num_samp_full by 
 *                           num_feat_full. Each row is a first moment (mean).
 *                           Must be initialized to 0.
 * */
{ 
    // Initialize the weight normalizer to 0.
    double normalizer_array[num_feat_full * num_samp_full];
    for (int64_t n = 0 ; n < num_feat_full*num_samp_full ; n++) {
        normalizer_array[n] = 0;
    }

    int64_t ind_samp_comp; //indexes the rows of compressed_array
    int64_t ind_samp_full; //indexes the rows of first_moment_array
    int64_t ind_feat_full; //indexes the columns of first_moment_array
    int64_t position_tracker; //indexes absolute position in first_moment_array

    // update the moments and the normalizer for each sample
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samp_comp ; ind_samp_comp++) {
        update_weighted_first_moment_array(first_moment_array,
                                           normalizer_array,
                                           &compressed_array[ind_samp_comp*num_feat_comp],
                                           &mask_array[ind_samp_comp*num_feat_comp],
                                           &weights_array[ind_samp_comp*num_samp_full],
                                           num_samp_full,
                                           num_feat_comp,
                                           num_feat_full);
    }
    // divide by the normalizer
    for (ind_samp_full = 0 ; ind_samp_full < num_samp_full ; ind_samp_full ++) {
        for (ind_feat_full = 0 ; ind_feat_full < num_feat_full ; ind_feat_full++) {
            position_tracker = ind_samp_full * num_feat_full + ind_feat_full;
            if (normalizer_array[position_tracker] > GREATER_THAN_ZERO_TOL) 
                first_moment_array[position_tracker] *= 1/normalizer_array[position_tracker];
            else
                first_moment_array[position_tracker] = 0;
        }
    }
}

void update_weighted_first_and_second_moment(double *first_moment_to_update,
                                       double *second_moment_to_update,
                                       double *normalizer_to_update,
                                       double *compressed_sample,
                                       int64_t *mask,
                                       double weight,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full)
/* Performs an update to the first and second moment using a single 
 * compressed_sample and weight. Also updates the normalizer. See
 * docs for update_weighted_first_moment for details. 
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     weight : the weight associated with compressed_sample
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 *
 * Returns
 * -------
 *
 *      first_moment_to_update : (modified) array, len num_feat_full. 
 *                               The current first moment (mean), to be 
 *                               updated.
 *
 *      second_moment_to_update : (modified) array, len num_feat_full. 
 *                               The current second moment, to be 
 *                               updated.
 *
 *      normalizer_to_update : (modified) array, len num_feat_full. The 
 *                             current normalizer, used to keep track of
 *                             which weights have been used in which 
 *                             dimensions.
 */
{ 
    int64_t ind_feat_comp = 0; //indexes through compressed_sample and mask
    int64_t ind_feat_full = 0; //indexes through moment and normalizer
    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp++ ) {
        ind_feat_full = mask[ind_feat_comp];
        first_moment_to_update[ind_feat_full] += weight * compressed_sample[ind_feat_comp];
        second_moment_to_update[ind_feat_full] += weight * \
            compressed_sample[ind_feat_comp] * compressed_sample[ind_feat_comp];
        normalizer_to_update[ind_feat_full] += weight;
    }
}

void update_weighted_first_and_second_moment_array(double *first_moment_array,
                                            double *second_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full)
/* Update a set of first moments using a single sample. Wrapper for
 * update_weighted_first_moment; see that function's docs for details.
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     weights : array, length num_samp_full, ith entry is the weight
 *               associated with compressed_sample for the ith moment. 
 *
 *     num_samp_full : the number of moments to be updated. 
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      first_moment_array : (modified) array, size num_samp_full by 
 *                           num_feat_full. Each row is a first moment (mean)
 *                           to be updated.
 *
 *      second_moment_array : (modified) array, size num_samp_full by 
 *                            num_feat_full. Each row is a second moment
 *                            to be updated.
 *
 *      normalizer_array : (modified) array, size num_samp_full by num_feat_full. 
 *                         Each row is a normalizer, used to keep track of
 *                         which weights have been used in which dimensions.
 * */
{ 
    int64_t ind_samp_full = 0; //indexes rows of first_moment_array
    for (ind_samp_full = 0 ; ind_samp_full < num_samp_full ; ind_samp_full ++) {
        update_weighted_first_and_second_moment(&first_moment_array[ind_samp_full*num_feat_full],
                                          &second_moment_array[ind_samp_full*num_feat_full],
                                          &normalizer_array[ind_samp_full*num_feat_full],
                                          compressed_sample,
                                          mask,
                                          weights[ind_samp_full],
                                          num_feat_comp,
                                          num_feat_full);
    }
}


void compute_weighted_first_and_second_moment_array(double *first_moment_array,
                                double *second_moment_array,
                                double *compressed_array,
                                int64_t *mask_array,
                                double *weights_array,
                                int64_t num_samp_comp,
                                int64_t num_samp_full,
                                int64_t num_feat_comp,
                                int64_t num_feat_full)
/* Compute weighted first moment and second. Every column of weights_array (and
 * correspondingly first_moment_array) corresponds to a set of weights.
 * For each set of weights, compute the sum of the samples in compressed_array
 * (the rows), weighted by the elements of this set of weights. 
 *
 * Wrapper for update_weighted_first_moment_array, calling this function
 * for each row of compressed_array. 
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the compressed domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the compressed sample in compressed_array was obtained.
 *
 *     weights_array : array, size num_samp_comp by num_samp_full.
 *                     ith col is the weights associated with each row of
 *                     compressed_sample for the ith moment. 
 *
 *     num_samp_comp : the number of samples (rows) in compressed_array
 *
 *     num_samp_full : the number of moments to be updated. 
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      first_moment_array : (modified) array, size num_samp_full by 
 *                           num_feat_full. Each row is a first moment (mean).
 *                           Must be initialized to 0.
 *
 *      second_moment_array : (modified) array, size num_samp_full by 
 *                           num_feat_full. Each row is a first moment (mean).
 *                           Must be initialized to 0.
 * */
{ 
    // Initialize the weight normalizer to 0.
    double normalizer_array[num_feat_full * num_samp_full];
    for (int64_t n = 0 ; n < num_feat_full*num_samp_full ; n++) {
        normalizer_array[n] = 0;
    }

    int64_t ind_samp_comp; //indexes the rows of compressed_array
    int64_t ind_samp_full; //indexes the rows of first_moment_array
    int64_t ind_feat_full; //indexes the columns of first_moment_array
    int64_t position_tracker; //indexes absolute position in first_moment_array

    // update the moments and the normalizer for each sample
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samp_comp ; ind_samp_comp++) {
        update_weighted_first_and_second_moment_array(first_moment_array,
                                                second_moment_array,
                                                normalizer_array,
                                                &compressed_array[ind_samp_comp*num_feat_comp],
                                                &mask_array[ind_samp_comp*num_feat_comp],
                                                &weights_array[ind_samp_comp*num_samp_full],
                                                num_samp_full,
                                                num_feat_comp,
                                                num_feat_full);
    }
    // divide by the normalizer
    for (ind_samp_full = 0 ; ind_samp_full < num_samp_full ; ind_samp_full ++) {
        for (ind_feat_full = 0 ; ind_feat_full < num_feat_full ; ind_feat_full++) {
            position_tracker = ind_samp_full * num_feat_full + ind_feat_full;
            if (normalizer_array[position_tracker] > GREATER_THAN_ZERO_TOL) {
                first_moment_array[position_tracker] *= 1/normalizer_array[position_tracker];
                second_moment_array[position_tracker] *= 1/normalizer_array[position_tracker];
            }
            else {
                first_moment_array[position_tracker] = 0;
                second_moment_array[position_tracker] = 0;
            }
        }
    }
}
