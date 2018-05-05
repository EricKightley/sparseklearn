#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "fastLA.h"

// rr : result row
// rc : result column
// dr : data row
// dc : data column
// nrow : number of rows in result
// ncol : number of columns in result
//
#define macro_U0(func_U0, inner_expr)\
int64_t func_U0(double *result, int64_t nrow, int64_t ncol, struct CONSTANTS *C, struct DATA *D,\
                int64_t *S, double *W, double *Sigma, int power)\
{\
    int64_t rr, rc, dr, dc, q1, q2, q_shared;\
    double scale;\
    for (rr = 0 ; rr < nrow - 1 ; rr++) {\
        dr = S[rr];\
        for (rc = rr + 1 ; rc < ncol ; rc++) {\
            dc = S[rc];\
            q1 = 0;\
            q2 = 0;\
            q_shared = 0;\
            while (q1 < C->Q && q2 < C->Q) {\
                if (D->mask[(C->Q)*dr+q1] < D->mask[(C->Q)*dc+q2])\
                    q1++;\
                else if (D->mask[(C->Q)*dr+q1] > D->mask[(C->Q)*dc+q2])\
                    q2++;\
                else {\
                    inner_expr\
                    q1++;\
                    q2++;\
                    q_shared++;\
                }\
            }\
            if (q_shared > 0) {\
                if (power == 1)\
                    scale = (float)C->P/(float)q_shared * (float)C->P/(float)q_shared;\
                else if (power == 2)\
                    scale = (float)C->P/(float)q_shared;\
                result[rr*ncol+rc] *= scale;\
            }\
        }\
    }\
    if (power == 2)\
        sqrt_array(result, nrow, ncol);\
    for (rr = 0 ; rr < ncol - 1 ; rr++) {\
        for (rc = rr + 1 ; rc < ncol  ; rc++) {\
            result[rc*ncol+rr] = result[rr*ncol+rc];\
        }\
    }\
}

#define macro_U1(func_U1, inner_expr)\
int64_t func_U1(double *result, int64_t nrow, int64_t ncol, struct CONSTANTS *C, struct DATA *D,\
                int64_t *S, double *W, double *U, double *Sigma, int power)\
{\
    int64_t rr, rc, dr, q, q_to_p;\
    double scale;\
    if (power == 1)\
        scale = ((float)C->P / (float)C->Q) * ((float)C->P / (float)C->Q);\
    else if (power == 2)\
        scale = (float)C->P / (float)C->Q;\
    for (rr = 0 ; rr < nrow ; rr++) {\
        for (rc = 0 ; rc < ncol ; rc++) {\
            dr = S[rc];\
            for (q = 0 ; q < C->Q ; q++) {\
                q_to_p = D->mask[dr*(C->Q) + q];\
                inner_expr\
            }\
            result[rr*ncol+rc] *= scale;\
        }\
    }\
    if (power == 2)\
       sqrt_array(result, nrow, ncol);\
}

macro_U0(
    _U0_W0_Sig0_Pow1,
    result[ncol*rr+rc] += fabs(D->RHDX[(C->Q)*dr+q1] - D->RHDX[(C->Q)*dc+q2]);
)

macro_U0(
    _U0_W0_Sig0_Pow2,
    result[ncol*rr+rc] += \
        (D->RHDX[(C->Q)*dr+q1] - D->RHDX[(C->Q)*dc+q2]) * \
        (D->RHDX[(C->Q)*dr+q1] - D->RHDX[(C->Q)*dc+q2]);
)

macro_U1(
    _U1_W0_Sig0_Pow1,
    result[ncol*rr+rc] += fabs(D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]);
)

macro_U1(
    _U1_W0_Sig0_Pow2,
    result[ncol*rr+rc] += (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) * \
        (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]);
)

macro_U1(
    _U1_W0_Sig1_Pow1,
    result[ncol*rr+rc] += fabs(D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) / \
        Sigma[(C->P)*rr+q_to_p];
)

macro_U1(
    _U1_W0_Sig1_Pow2,
    result[ncol*rr+rc] += (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) * \
        (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) / Sigma[(C->P)*rr+q_to_p];
)

macro_U1(
    _U1_W1_Sig0_Pow1,
    result[ncol*rr+rc] += W[nrow*dr+rr] * fabs(D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]);
)

macro_U1(
    _U1_W1_Sig0_Pow2,
    result[ncol*rr+rc] += W[nrow*dr+rr] * (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) * \
        (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]);
)

macro_U1(
    _U1_W1_Sig1_Pow1,
    result[ncol*rr+rc] += W[nrow*dr+rr] * fabs(D->RHDX[(C->Q)*dr+q] - \
        U[(C->P)*rr+q_to_p]) / Sigma[(C->P)*rr+q_to_p];
)

macro_U1(
    _U1_W1_Sig1_Pow2,
    result[ncol*rr+rc] += W[nrow*dr+rr] * (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) * \
        (D->RHDX[(C->Q)*dr+q] - U[(C->P)*rr+q_to_p]) / Sigma[(C->P)*rr+q_to_p];
)

void sqrt_array(double *array, const int64_t nrow, const int64_t ncol)
{
    int64_t r,c;
    for (r = 0 ; r < nrow ; r++) {
        for (c = 0 ; c < ncol ; c++) {
            array[r*ncol + c] = sqrt(array[r*ncol + c]);
        }
    }
}

void pwdist(double *result, int64_t nrow, int64_t ncol, struct CONSTANTS *C, struct DATA *D, 
            int power, int64_t *S, double *W, double *U, double *Sigma)
{
    if (!U && !W && !Sigma && power == 1)
        _U0_W0_Sig0_Pow1(result, nrow, ncol, C, D, S, W, Sigma, power);
    else if (!U && !W && !Sigma && power == 2)
        _U0_W0_Sig0_Pow2(result, nrow, ncol, C, D, S, W, Sigma, power);
    else if ( U && !W && !Sigma && power == 1)
        _U1_W0_Sig0_Pow1(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U && !W && !Sigma && power == 2)
        _U1_W0_Sig0_Pow2(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U && !W &&  Sigma && power == 1)
        _U1_W0_Sig1_Pow1(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U && !W &&  Sigma && power == 2)
        _U1_W0_Sig1_Pow2(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U &&  W && !Sigma && power == 1)
        _U1_W1_Sig0_Pow1(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U &&  W && !Sigma && power == 2)
        _U1_W1_Sig0_Pow2(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U &&  W &&  Sigma && power == 1)
        _U1_W1_Sig1_Pow1(result, nrow, ncol, C, D, S, W, U, Sigma, power);
    else if ( U &&  W &&  Sigma && power == 2)
        _U1_W1_Sig1_Pow2(result, nrow, ncol, C, D, S, W, U, Sigma, power);
}
