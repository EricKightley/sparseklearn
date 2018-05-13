#include <stdio.h>
#include <stdint.h>
#include <math.h>

void apply_mask_to_full_sample(double *result,
                               double *samp_full, 
                               int64_t *mask, 
                               int64_t num_feat_comp);
