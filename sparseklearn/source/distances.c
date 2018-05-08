#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "distances.h"

double dist_both_comp(double *comp_sample_1, 
                                double *comp_sample_2, 
                                int64_t *mask_1, 
                                int64_t *mask_2,
                                int64_t num_feat_comp,
                                int64_t num_feat_full)
/* Computes the l2 distance between comp_sample_1 and 
 * comp_sample_2, by finding the intersection of their masks, computing
 * the distance between the samples projected onto this common subspace, and
 * then scaling this distance back up.
 *
 * Inputs
 * ------
 *
 *     comp_sample_1 : array, length num_feat_comp
 *
 *     comp_sample_2 : array, length num_feat_comp
 *
 *     mask_1 : array, length num_feat_comp. The indices specifying which 
 *              entries of the full sample_1 were kept. Must be sorted.
 *
 *     mask_2 : array, length num_feat_comp. The indices specifying which 
 *              entries of the full sample_2 were kept. Must be sorted.
 *
 *     num_feat_comp : the number of features in a comp sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate l2 distance between both samples.
 */
{
    int64_t feat_ind_1 = 0; //iterates over entries of comp_sample_1
    int64_t feat_ind_2 = 0; //iterates over entries of comp_sample_2
    int64_t mask_intersection_size = 0; //how many mask entries are shared
    double distance = 0;

    // find the intersection of the sorted masks
    while (feat_ind_1 < num_feat_comp && feat_ind_2 < num_feat_comp) {
        if (mask_1[feat_ind_1] < mask_2[feat_ind_2])
            feat_ind_1++;
        else if (mask_1[feat_ind_1] > mask_2[feat_ind_2])
            feat_ind_2++;
        else { //this mask index is shared by both samples
            distance += ( comp_sample_1[feat_ind_1] -   \
                          comp_sample_2[feat_ind_2] ) * \
                        ( comp_sample_1[feat_ind_1] -   \
                          comp_sample_2[feat_ind_2] );
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


double dist_one_comp_one_full(double *comp_sample, 
                                        double *full_sample, 
                                        int64_t *mask, 
                                        int64_t num_feat_comp,
                                        int64_t num_feat_full)
/* Computes the l2 distance between comp_sample_1 and 
 * comp_sample_2, by finding the intersection of their masks, computing
 * the distance between the samples projected onto this common subspace, and
 * then scaling this distance back up.
 *
 * Inputs
 * ------
 *
 *     comp_sample : array, length num_feat_comp
 *
 *     full_sample : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted.
 *
 *     num_feat_comp : the number of features in a comp sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate l2 distance between both samples.
 */
{
    int64_t ind_comp = 0; //indexes entries of comp_sample
    double distance = 0;

    for (ind_comp = 0 ; ind_comp < num_feat_comp ; ind_comp ++) {
        distance += ( comp_sample[ind_comp] -   \
                      full_sample[mask[ind_comp]] ) * \
                    ( comp_sample[ind_comp] -   \
                      full_sample[mask[ind_comp]] );
    }

    distance = sqrt(distance);

    // rescale the distance from the comp dimension to the full dimension
    distance *= sqrt((float)num_feat_full / (float)num_feat_comp);

    return distance;
}


void pairwise_l2_distances_with_self(double *result,
                                     double *comp_array,
                                     int64_t *mask_array,
                                     int64_t num_samples,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/*  Computes the pairwise l2 distance between the rows of comp_array by 
 *  intersecting the the masks of each pair of datapoints (x,y) and
 *  approximating ||x-y||_2 in the full domain with ||x-y||_2 * scale
 *  in the reduced domain specified by where x and y have common entries. 
 *
 * Inputs
 * ------
 *
 *     comp_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the comp domain.
 *
 *     mask : array, size num_samples by num_feat_comp. Each row is the indices 
 *            indicating which entries of the full datapoint were kept. 
 *
 *     num_samples : the number of samples (rows) in comp_array
 *
 *     num_feat_comp : the number of dimensions in the comp data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              distances between each datapoint; i.e., result[i][j] is the 
 *              distance between the ith and jth row of comp_array. 
 */ 
{
    int64_t ind_samp1 = 0; //indexes rows of comp_array
    int64_t ind_samp2 = 0; //indexes rows of comp_array
    // upper triangular 
    for (ind_samp1 = 0 ; ind_samp1 < num_samples - 1 ; ind_samp1++) {
        for (ind_samp2 = ind_samp1 + 1 ; ind_samp2 < num_samples ; ind_samp2++) {
            result[ind_samp1 * num_samples + ind_samp2] = \
            dist_both_comp(&comp_array[ind_samp1*num_feat_comp],
                                         &comp_array[ind_samp2*num_feat_comp],
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
                                     double *comp_array,
                                     double *full_array,
                                     int64_t *mask_array,
                                     int64_t num_samples_comp,
                                     int64_t num_samples_full,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/*  Computes the pairwise l2 distance between the rows of comp_array and 
 *  the rows of full_array by masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the comp domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     comp_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the comp domain.
 *
 *     full_array : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask : array, size num_samples by num_feat_comp. Each row is the indices 
 *            indicating which entries were kept of the full datapoint from 
 *            which the comp sample in comp_array was obtained.
 *
 *     num_samples_comp : the number of samples (rows) in comp_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the comp data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              distances between each datapoint; i.e., result[i][j] is the 
 *              distance between the ith and jth row of comp_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of comp_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            dist_one_comp_one_full(&comp_array[ind_samp_comp*num_feat_comp],
                                                 &full_array[ind_samp_full*num_feat_full],
                                                 &mask_array[ind_samp_comp*num_feat_comp],
                                                 num_feat_comp,
                                                 num_feat_full);
        }
    }
}


double mahalanobis_distance_spherical(double *comp_sample,
                                      double *full_mean,
                                      int64_t *mask,
                                      double spherical_covariance,
                                      int64_t num_feat_comp,
                                      int64_t num_feat_full)
/* Computes the Mahalanobis distance between comp_sample and full_mean 
 * by projecting full_mean into the comp domain using comp_sample's
 * mask, then scaling this distance back up
 *
 * Inputs
 * ------
 *
 *     comp_sample : array, length num_feat_comp
 *
 *     full_mean : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     spherical_covariance : spherical variance (sigma^2)
 *
 *     num_feat_comp : the number of features in a comp sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate mahalanobis distance between 
 *                 comp_sample and full_mean using spherical_covariance.
 */
{
    int64_t ind_feat_comp = 0; //indexes entries of comp_sample
    double distance = 0;

    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp ++) {
        distance += ( comp_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) * \
                    ( comp_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] );
    }
    // divide by variance
    distance *= 1/spherical_covariance;
    // rescale from comp dimension to full
    distance *= (float)num_feat_full / (float)num_feat_comp;
    // all of this needs to be sqrt
    distance = sqrt(distance);
    return distance;
}


double mahalanobis_distance_diagonal(double *comp_sample,
                                     double *full_mean,
                                     int64_t *mask,
                                     double *diagonal_covariance,
                                     int64_t num_feat_comp,
                                     int64_t num_feat_full)
/* Computes the Mahalanobis distance between comp_sample and full_mean 
 * by projecting full_mean into the comp domain using comp_sample's
 * mask, then scaling this distance back up
 *
 * Inputs
 * ------
 *
 *     comp_sample : array, length num_feat_comp
 *
 *     full_mean : array, length num_feat_full
 *
 *     mask : array, length num_feat_comp. The indices specifying which 
 *                   entries of the full sample_1 were kept. Must be sorted
 *
 *     diagonal_covariance : array, length num_feat_full. 
 *
 *     num_feat_comp : the number of features in a comp sample. 
 *
 *     num_feat_full : the number of features in a full sample. 
 *
 * Returns
 * -------
 *
 *      distance : double, the approximate mahalanobis distance between 
 *                 comp_sample and full_mean using spherical_covariance.
 */
{
    int64_t ind_feat_comp = 0; //indexes entries of comp_sample
    double distance = 0;

    for (ind_feat_comp = 0 ; ind_feat_comp < num_feat_comp ; ind_feat_comp ++) {
        distance += ( comp_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) * \
                    ( comp_sample[ind_feat_comp] -   \
                      full_mean[mask[ind_feat_comp]] ) / \
                    diagonal_covariance[mask[ind_feat_comp]];
    }
    // rescale from comp dimension to full
    distance *= (float)num_feat_full / (float)num_feat_comp;
    // all of this needs to be sqrt
    distance = sqrt(distance);
    return distance;
}


void pairwise_mahalanobis_distances_spherical(double *result,
                                             double *comp_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *spherical_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full)
/*  Computes the pairwise spherical mahalanobis distance between 
 *  the rows of comp_array and the rows of full_array by 
 *  masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the comp domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     comp_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the comp domain.
 *
 *     full_means : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the comp sample in comp_array was obtained.
 *
 *     spherical_covariance_array : array, size num_samples_full. Each entry is
 *                                  the spherical covariance for the corresponding
 *                                  mean.              
 *
 *     num_samples_comp : the number of samples (rows) in comp_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the comp data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              mahalanobis distance between each datapoint; i.e., result[i][j] 
 *              is the distance between the ith and jth row of comp_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of comp_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            mahalanobis_distance_spherical(&comp_array[ind_samp_comp*num_feat_comp],
                                           &full_means[ind_samp_full*num_feat_full],
                                           &mask_array[ind_samp_comp*num_feat_comp],
                                           spherical_covariance_array[ind_samp_full],
                                           num_feat_comp,
                                           num_feat_full);
        }
    }
}


void pairwise_mahalanobis_distances_diagonal(double *result,
                                             double *comp_array,
                                             double *full_means,
                                             int64_t *mask_array,
                                             double *diagonal_covariance_array,
                                             int64_t num_samples_comp,
                                             int64_t num_samples_full,
                                             int64_t num_feat_comp,
                                             int64_t num_feat_full)
/*  Computes the pairwise mahalanobis distances, with diagonal covariances, 
 *  between the rows of comp_array and the rows of full_array by 
 *  masking each sample in full_array with each mask 
 *  in mask_array, finding the l2 distance in the comp domain, and then
 *  scaling back up to approximate the distance in the full domain.
 *
 * Inputs
 * ------
 *
 *     comp_array : array, size num_samples by num_feat_comp. Each row is
 *                        a datapoint in the comp domain.
 *
 *     full_means : array, size num_samples by num_feat_full. Each row is
 *                  a datapoint in the full domain.
 *
 *     mask_array : array, size num_samples by num_feat_comp. Each row is the indices 
 *                  indicating which entries were kept of the full datapoint from 
 *                  which the comp sample in comp_array was obtained.
 *
 *     diagonal_covariance_array : array, size num_samples_full by num_feat_full. 
 *                                 Each row is the diagonal covariance for the 
 *                                 corresponding mean.              
 *
 *     num_samples_comp : the number of samples (rows) in comp_array
 *
 *     num_samples_full : the number of samples (rows) in full_array
 *
 *     num_feat_comp : the number of dimensions in the comp data
 *
 *     num_feat_full : the number of dimensions in the full data
 *
 * Returns
 * -------
 *
 *     result : (modified) array, size num_samples by num_samples, the pairwise
 *              mahalanobis distance between each datapoint; i.e., result[i][j] 
 *              is the distance between the ith and jth row of comp_array. 
 */ 
{
    int64_t ind_samp_comp = 0; //indexes rows of comp_array
    int64_t ind_samp_full = 0; //indexes rows of full_array
    for (ind_samp_comp = 0 ; ind_samp_comp < num_samples_comp ; ind_samp_comp ++) {
        for (ind_samp_full = 0 ; ind_samp_full < num_samples_full ; ind_samp_full ++) {
            result[ind_samp_comp*num_samples_full+ind_samp_full] = \
            mahalanobis_distance_diagonal(&comp_array[ind_samp_comp*num_feat_comp],
                                           &full_means[ind_samp_full*num_feat_full],
                                           &mask_array[ind_samp_comp*num_feat_comp],
                                           &diagonal_covariance_array[ind_samp_full*num_feat_full],
                                           num_feat_comp,
                                           num_feat_full);
        }
    }
}




