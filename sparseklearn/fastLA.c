#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "fastLA.h"


double _l2_distance_both_compressed(double *compressed_sample_1, 
                                double *compressed_sample_2, 
                                int64_t *mask_1, 
                                int64_t *mask_2,
                                int64_t num_feat_comp,
                                int64_t num_feat_full)
/* Computes the l2 distance between compressed_sample_1 and 
 * compressed_sample_2, by finding the intersection of their masks, computing
 * the distance between the samples projected onto this common subspace, and
 * then scaling this distance back up.
 *
 * Inputs
 * ------
 *
 *     compressed_sample_1 : array, length num_feat_comp
 *
 *     compressed_sample_2 : array, length num_feat_comp
 *
 *     mask_1 : array, length num_feat_comp. The indices specifying which 
 *              entries of the full sample_1 were kept. Must be sorted.
 *
 *     mask_2 : array, length num_feat_comp. The indices specifying which 
 *              entries of the full sample_2 were kept. Must be sorted.
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate l2 distance between both samples.
 */
{
    int64_t feat_ind_1 = 0; //iterates over entries of compressed_sample_1
    int64_t feat_ind_2 = 0; //iterates over entries of compressed_sample_2
    int64_t mask_intersection_size = 0; //how many mask entries are shared
    double distance = 0;

    // find the intersection of the sorted masks
    while (feat_ind_1 < num_feat_comp && feat_ind_2 < num_feat_comp) {
        if (mask_1[feat_ind_1] < mask_2[feat_ind_2])
            feat_ind_1++;
        else if (mask_1[feat_ind_1] > mask_2[feat_ind_2])
            feat_ind_2++;
        else { //this mask index is shared by both samples
            distance += ( compressed_sample_1[feat_ind_1] -   \
                          compressed_sample_2[feat_ind_2] ) * \
                        ( compressed_sample_1[feat_ind_1] -   \
                          compressed_sample_2[feat_ind_2] );
            feat_ind_1++;
            feat_ind_2++;
            mask_intersection_size++;
        }
    }

    distance = sqrt(distance);

    // rescale the distance from mask_intersection_size to the full dimension
    if (mask_intersection_size > 0) {
        distance *= sqrt((float)num_feat_full / (float)mask_intersection_size);
    }

    return distance;
}


double _l2_distance_one_compressed_one_full(double *compressed_sample, 
                                        double *full_sample, 
                                        int64_t *mask, 
                                        int64_t num_feat_comp,
                                        int64_t num_feat_full)
/* Computes the l2 distance between compressed_sample_1 and 
 * compressed_sample_2, by finding the intersection of their masks, computing
 * the distance between the samples projected onto this common subspace, and
 * then scaling this distance back up.
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     full_sample : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted.
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate l2 distance between both samples.
 */
{
    int64_t ind_comp = 0; //indexes entries of compressed_sample
    double distance = 0;

    for (ind_comp = 0 ; ind_comp < num_feat_comp ; ind_comp ++) {
        distance += ( compressed_sample[ind_comp] -   \
                      full_sample[mask[ind_comp]] ) * \
                    ( compressed_sample[ind_comp] -   \
                      full_sample[mask[ind_comp]] );
    }

    distance = sqrt(distance);

    // rescale the distance from the compressed dimension to the full dimension
    distance *= sqrt((float)num_feat_full / (float)num_feat_comp);

    return distance;
}


void pairwise_l2_distances_with_self(double *result,
                                     double *compressed_array,
                                     int64_t *mask_array,
                                     int64_t num_samples,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/*  Computes the pairwise l2 distance between the rows of compressed_array by 
 *  intersecting the the masks of each pair of datapoints (x,y) and
 *  approximating ||x-y||_2 in the full domain with ||x-y||_2 * scale
 *  in the reduced domain specified by where x and y have common entries. 
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the compressed domain.
 *
 *     mask : array, size num_samples by num_feat_comp. Each row is the indices 
 *            indicating which entries of the full datapoint were kept. 
 *
 *     num_samples : the number of samples (rows) in compressed_array
 *
 *     num_feat_comp : the number of dimensions in the compressed data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              distances between each datapoint; i.e., result[i][j] is the 
 *              distance between the ith and jth row of compressed_array. 
 */ 
{
    int64_t ind_samp1 = 0; //indexes rows of compressed_array
    int64_t ind_samp2 = 0; //indexes rows of compressed_array
    // upper triangular 
    for (ind_samp1 = 0 ; ind_samp1 < num_samples - 1 ; ind_samp1++) {
        for (ind_samp2 = ind_samp1 + 1 ; ind_samp2 < num_samples ; ind_samp2++) {
            result[ind_samp1 * num_samples + ind_samp2] = \
            _l2_distance_both_compressed(&compressed_array[ind_samp1*num_feat_comp],
                                         &compressed_array[ind_samp2*num_feat_comp],
                                         &mask_array[ind_samp1*num_feat_comp],
                                         &mask_array[ind_samp2*num_feat_comp],
                                         num_feat_comp,
                                         num_feat_full);
        }
    }
    // lower triangular and diagonal
    for (ind_samp1 = 0 ; ind_samp1 < num_samples ; ind_samp1++) {
        //the diagonal is 0
        result[ind_samp1 * num_samples + ind_samp1] = 0;
        //the lower triangular is copied from the upper
        for (ind_samp2 = 0 ; ind_samp2 < ind_samp1 ; ind_samp2++) {
            result[ind_samp1 * num_samples + ind_samp2] = \
            result[ind_samp2 * num_samples + ind_samp1];
        }
    }
}


void pairwise_l2_distances_with_full(double *result,
                                     double *compressed_array,
                                     double *full_array,
                                     int64_t *mask_array,
                                     int64_t num_samples_comp,
                                     int64_t num_samples_full,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/*  Computes the pairwise l2 distance between the rows of compressed_array and 
 *  the rows of full_array by masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the compressed domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the compressed domain.
 *
 *     full_array : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask : array, size num_samples by num_feat_comp. Each row is the indices 
 *            indicating which entries were kept of the full datapoint from 
 *            which the compressed sample in compressed_array was obtained.
 *
 *     num_samples_comp : the number of samples (rows) in compressed_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the compressed data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              distances between each datapoint; i.e., result[i][j] is the 
 *              distance between the ith and jth row of compressed_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of compressed_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            _l2_distance_one_compressed_one_full(&compressed_array[ind_samp_comp*num_feat_comp],
                                                 &full_array[ind_samp_full*num_feat_full],
                                                 &mask_array[ind_samp_comp*num_feat_comp],
                                                 num_feat_comp,
                                                 num_feat_full);
        }
    }
}


double mahalanobis_distance_spherical(double *compressed_sample,
                                      double *full_mean,
                                      int64_t *mask,
                                      double spherical_covariance,
                                      int64_t num_feat_comp,
                                      int64_t num_feat_full)
/* Computes the Mahalanobis distance between compressed_sample and full_mean 
 * by projecting full_mean into the compressed domain using compressed_sample's
 * mask, then scaling this distance back up
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     full_mean : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     spherical_covariance : spherical variance (sigma^2)
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate mahalanobis distance between 
 *                 compressed_sample and full_mean using spherical_covariance.
 */
{
    int64_t ind_feat_comp = 0; //indexes entries of compressed_sample
    double distance = 0;

    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp ++) {
        distance += ( compressed_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) * \
                    ( compressed_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] );
    }
    // divide by variance
    distance *= 1/spherical_covariance;
    // rescale from compressed dimension to full
    distance *= (float)num_feat_full / (float)num_feat_comp;
    // all of this needs to be sqrt
    distance = sqrt(distance);
    return distance;
}


double mahalanobis_distance_diagonal(double *compressed_sample,
                                     double *full_mean,
                                     int64_t *mask,
                                     double *diagonal_covariance,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/* Computes the Mahalanobis distance between compressed_sample and full_mean 
 * by projecting full_mean into the compressed domain using compressed_sample's
 * mask, then scaling this distance back up
 *
 * Inputs
 * ------
 *
 *     compressed_sample : array, length num_feat_comp
 *
 *     full_mean : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     diagonal_covariance : array, length num_feat_full. 
 *
 *     num_feat_comp : the number of features in a compressed sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate mahalanobis distance between 
 *                 compressed_sample and full_mean using spherical_covariance.
 */
{
    int64_t ind_feat_comp = 0; //indexes entries of compressed_sample
    double distance = 0;

    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp ++) {
        distance += ( compressed_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) * \
                    ( compressed_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) / \
                    diagonal_covariance[mask[ind_feat_comp]];
    }
    // rescale from compressed dimension to full
    distance *= (float)num_feat_full / (float)num_feat_comp;
    // all of this needs to be sqrt
    distance = sqrt(distance);
    return distance;
}


void pairwise_mahalanobis_distances_spherical(double *result,
                                             double *compressed_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *spherical_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full)
/*  Computes the pairwise spherical mahalanobis distance between 
 *  the rows of compressed_array and the rows of full_array by 
 *  masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the compressed domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the compressed domain.
 *
 *     full_means : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the compressed sample in compressed_array was obtained.
 *
 *     spherical_covariance_array : array, size num_samples_full. Each entry is
 *                                  the spherical covariance for the corresponding
 *                                  mean.              
 *
 *     num_samples_comp : the number of samples (rows) in compressed_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the compressed data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              mahalanobis distance between each datapoint; i.e., result[i][j] 
 *              is the distance between the ith and jth row of compressed_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of compressed_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            mahalanobis_distance_spherical(&compressed_array[ind_samp_comp*num_feat_comp],
                                           &full_means[ind_samp_full*num_feat_full],
                                           &mask_array[ind_samp_comp*num_feat_comp],
                                           spherical_covariance_array[ind_samp_full],
                                           num_feat_comp,
                                           num_feat_full);
        }
    }
}


void pairwise_mahalanobis_distances_diagonal(double *result,
                                             double *compressed_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *diagonal_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full)
/*  Computes the pairwise mahalanobis distances, with diagonal covariances, 
 *  between the rows of compressed_array and the rows of full_array by 
 *  masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the compressed domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     compressed_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the compressed domain.
 *
 *     full_means : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the compressed sample in compressed_array was obtained.
 *
 *     diagonal_covariance_array : array, size num_samples_full by num_feat_full. 
 *                                 Each row is the diagonal covariance for the 
 *                                 corresponding mean.              
 *
 *     num_samples_comp : the number of samples (rows) in compressed_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the compressed data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              mahalanobis distance between each datapoint; i.e., result[i][j] 
 *              is the distance between the ith and jth row of compressed_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of compressed_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            mahalanobis_distance_diagonal(&compressed_array[ind_samp_comp*num_feat_comp],
                                           &full_means[ind_samp_full*num_feat_full],
                                           &mask_array[ind_samp_comp*num_feat_comp],
                                           &diagonal_covariance_array[ind_samp_full*num_feat_full],
                                           num_feat_comp,
                                           num_feat_full);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//  First and Second Moments

void update_first_moment_single_sample(double *first_moment_to_update,
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


void update_both_moments_single_sample(double *first_moment_to_update,
                                       double *second_moment_to_update,
                                       double *normalizer_to_update,
                                       double *compressed_sample,
                                       int64_t *mask,
                                       double weight,
                                       int64_t num_feat_comp,
                                       int64_t num_feat_full)
/* Performs an update to the first and second moment using a single 
 * compressed_sample and weight. Also updates the normalizer. See
 * docs for update_first_moment_single_sample for details. 
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

void update_first_moment_array_single_sample(double *first_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full)
/* Update a set of first moments using a single sample. Wrapper for
 * update_first_moment_single_sample; see that function's docs for details.
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
        update_first_moment_single_sample(&first_moment_array[ind_samp_full*num_feat_full],
                                          &normalizer_array[ind_samp_full*num_feat_full],
                                          compressed_sample,
                                          mask,
                                          weights[ind_samp_full],
                                          num_feat_comp,
                                          num_feat_full);
    }
}

void update_both_moment_arrays_single_sample(double *first_moment_array,
                                            double *second_moment_array,
                                            double *normalizer_array,
                                            double *compressed_sample,
                                            int64_t *mask,
                                            double *weights,
                                            int64_t num_samp_full,
                                            int64_t num_feat_comp,
                                            int64_t num_feat_full)
/* Update a set of first moments using a single sample. Wrapper for
 * update_first_moment_single_sample; see that function's docs for details.
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
        update_both_moments_single_sample(&first_moment_array[ind_samp_full*num_feat_full],
                                          &second_moment_array[ind_samp_full*num_feat_full],
                                          &normalizer_array[ind_samp_full*num_feat_full],
                                          compressed_sample,
                                          mask,
                                          weights[ind_samp_full],
                                          num_feat_comp,
                                          num_feat_full);
    }
}


// U_ind : result row
// S_ind : result column
// X_ind : data row
// dc : data column
// num_samples_U : number of rows in result
// num_subsamples_X : number of columns in result
//
#define macro_U0(func_U0, inner_expr)\
int64_t func_U0(double *result, int64_t num_samples_U, int64_t num_subsamples_X, struct CONSTANTS *C, struct DATA *D,\
                int64_t *S, double *W, double *Sigma, int power)\
{\
    int64_t U_ind, S_ind, X_ind, dc, q1, q2, q_shared;\
    double scale;\
    for (U_ind = 0 ; U_ind < num_samples_U - 1 ; U_ind++) {\
        X_ind = S[U_ind];\
        for (S_ind = U_ind + 1 ; S_ind < num_subsamples_X ; S_ind++) {\
            dc = S[S_ind];\
            q1 = 0;\
            q2 = 0;\
            q_shared = 0;\
            while (q1 < C->Q && q2 < C->Q) {\
                if (D->mask[(C->Q)*X_ind+q1] < D->mask[(C->Q)*dc+q2])\
                    q1++;\
                else if (D->mask[(C->Q)*X_ind+q1] > D->mask[(C->Q)*dc+q2])\
                    q2++;\
                else {\
                    inner_expr\
                    q1++;\
                    q2++;\
                    q_shared++;\
                }\
            }\
            if (q_shared > 0) {\
                scale = (float)C->P/(float)q_shared;\
                result[U_ind*num_subsamples_X+S_ind] *= scale;\
            }\
        }\
    }\
    if (power == 2)\
        sqrt_array(result, num_samples_U, num_subsamples_X);\
    for (U_ind = 0 ; U_ind < num_subsamples_X - 1 ; U_ind++) {\
        for (S_ind = U_ind + 1 ; S_ind < num_subsamples_X  ; S_ind++) {\
            result[S_ind*num_subsamples_X+U_ind] = result[U_ind*num_subsamples_X+S_ind];\
        }\
    }\
}

#define macro_U1(func_U1, inner_expr)\
int64_t func_U1(double *result, int64_t num_samples_U, int64_t num_subsamples_X, struct CONSTANTS *C, struct DATA *D,\
                int64_t *S, double *W, double *U, double *Sigma, int power)\
{\
    int64_t U_ind, S_ind, X_ind, q, q_to_p;\
    double scale;\
    if (power == 1)\
        scale = ((float)C->P / (float)C->Q) * ((float)C->P / (float)C->Q);\
    else if (power == 2)\
        scale = (float)C->P / (float)C->Q;\
    for (U_ind = 0 ; U_ind < num_samples_U ; U_ind++) {\
        for (S_ind = 0 ; S_ind < num_subsamples_X ; S_ind++) {\
            X_ind = S[S_ind];\
            for (q = 0 ; q < C->Q ; q++) {\
                q_to_p = D->mask[X_ind*(C->Q) + q];\
                inner_expr\
            }\
            result[U_ind*num_subsamples_X+S_ind] *= scale;\
        }\
    }\
    if (power == 2)\
       sqrt_array(result, num_samples_U, num_subsamples_X);\
}

macro_U0(
    _U0_W0_Sig0_Pow1,
    result[num_subsamples_X*U_ind+S_ind] += fabs(D->RHDX[(C->Q)*X_ind+q1] - D->RHDX[(C->Q)*dc+q2]);
)

macro_U0(
    _U0_W0_Sig0_Pow2,
    result[num_subsamples_X*U_ind+S_ind] += \
        (D->RHDX[(C->Q)*X_ind+q1] - D->RHDX[(C->Q)*dc+q2]) * \
        (D->RHDX[(C->Q)*X_ind+q1] - D->RHDX[(C->Q)*dc+q2]);
)

macro_U1(
    _U1_W0_Sig0_Pow1,
    result[num_subsamples_X*U_ind+S_ind] += fabs(D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]);
)

macro_U1(
    _U1_W0_Sig0_Pow2,
    result[num_subsamples_X*U_ind+S_ind] += (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) * \
        (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]);
)

macro_U1(
    _U1_W0_Sig1_Pow1,
    result[num_subsamples_X*U_ind+S_ind] += fabs(D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) / \
        Sigma[(C->P)*U_ind+q_to_p];
)

macro_U1(
    _U1_W0_Sig1_Pow2,
    result[num_subsamples_X*U_ind+S_ind] += (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) * \
        (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) / Sigma[(C->P)*U_ind+q_to_p];
)

macro_U1(
    _U1_W1_Sig0_Pow1,
    result[num_subsamples_X*U_ind+S_ind] += W[num_samples_U*X_ind+U_ind] * fabs(D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]);
)

macro_U1(
    _U1_W1_Sig0_Pow2,
    result[num_subsamples_X*U_ind+S_ind] += W[num_samples_U*X_ind+U_ind] * (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) * \
        (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]);
)

macro_U1(
    _U1_W1_Sig1_Pow1,
    result[num_subsamples_X*U_ind+S_ind] += W[num_samples_U*X_ind+U_ind] * fabs(D->RHDX[(C->Q)*X_ind+q] - \
        U[(C->P)*U_ind+q_to_p]) / Sigma[(C->P)*U_ind+q_to_p];
)

macro_U1(
    _U1_W1_Sig1_Pow2,
    result[num_subsamples_X*U_ind+S_ind] += W[num_samples_U*X_ind+U_ind] * (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) * \
        (D->RHDX[(C->Q)*X_ind+q] - U[(C->P)*U_ind+q_to_p]) / Sigma[(C->P)*U_ind+q_to_p];
)

void sqrt_array(double *aU_inday, const int64_t num_samples_U, const int64_t num_subsamples_X)
{
    int64_t r,c;
    for (r = 0 ; r < num_samples_U ; r++) {
        for (c = 0 ; c < num_subsamples_X ; c++) {
            aU_inday[r*num_subsamples_X + c] = sqrt(aU_inday[r*num_subsamples_X + c]);
        }
    }
}

void pwdist(double *result, int64_t num_samples_U, int64_t num_subsamples_X, struct CONSTANTS *C, struct DATA *D, 
            int power, int64_t *S, double *W, double *U, double *Sigma)
{
    if (!U && !W && !Sigma && power == 1)
        _U0_W0_Sig0_Pow1(result, num_samples_U, num_subsamples_X, C, D, S, W, Sigma, power);
    else if (!U && !W && !Sigma && power == 2)
        _U0_W0_Sig0_Pow2(result, num_samples_U, num_subsamples_X, C, D, S, W, Sigma, power);
    else if ( U && !W && !Sigma && power == 1)
        _U1_W0_Sig0_Pow1(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U && !W && !Sigma && power == 2)
        _U1_W0_Sig0_Pow2(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U && !W &&  Sigma && power == 1)
        _U1_W0_Sig1_Pow1(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U && !W &&  Sigma && power == 2)
        _U1_W0_Sig1_Pow2(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U &&  W && !Sigma && power == 1)
        _U1_W1_Sig0_Pow1(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U &&  W && !Sigma && power == 2)
        _U1_W1_Sig0_Pow2(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U &&  W &&  Sigma && power == 1)
        _U1_W1_Sig1_Pow1(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
    else if ( U &&  W &&  Sigma && power == 2)
        _U1_W1_Sig1_Pow2(result, num_samples_U, num_subsamples_X, C, D, S, W, U, Sigma, power);
}
