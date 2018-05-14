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

