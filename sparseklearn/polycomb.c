#include <stdio.h>
#include <stdint.h>
#include <math.h>

// define general macro that we will use to create all of our more specific functions
#define funcdefine(func_name, inner_expr)\
int64_t func_name(double *result, const double *X, const int64_t* mask, const int64_t* S, const double* W,\
    const double* U, const double* Sigma, int64_t power, int64_t S_len, int64_t Q, int64_t N, int64_t P, int64_t K)\
{\
    for (int64_t i = 0; i < S_len; i++)\
    {\
        int64_t n = S[i];\
        for (int64_t q = 0; q < Q; q++)\
        {\
            int64_t q_to_p = mask[Q*n+q];\
            for (int64_t k = 0; k < K; k++)\
            {\
                inner_expr\
            }\
        }\
    }\
}

// define functions that ignore unnecessary operations in innermost loop
funcdefine(
    _noW_noU_noSigma_noPower,
    result[P*k+q_to_p] += X[Q*n+q];
)
funcdefine(
    _noW_noU_noSigma_yesPower,
    result[P*k+q_to_p] += pow(X[Q*n+q],power);
)
funcdefine(
    _noW_noU_yesSigma_noPower,
    result[P*k+q_to_p] += X[Q*n+q]/Sigma[P*k+q_to_p];
)
funcdefine(
    _noW_noU_yesSigma_yesPower,
    result[P*k+q_to_p] += pow(X[Q*n+q],power)/Sigma[P*k+q_to_p];
)
funcdefine(
    _noW_yesU_noSigma_noPower,
    result[P*k+q_to_p] += X[Q*n+q]-U[P*k+q_to_p];
)
funcdefine(
    _noW_yesU_noSigma_yesPower,
    result[P*k+q_to_p] += pow(X[Q*n+q]-U[P*k+q_to_p],power);
)
funcdefine(
    _noW_yesU_yesSigma_noPower,
    result[P*k+q_to_p] += (X[Q*n+q]-U[P*k+q_to_p])/Sigma[P*k+q_to_p];
)
funcdefine(
    _noW_yesU_yesSigma_yesPower,
    result[P*k+q_to_p] += pow(X[Q*n+q]-U[P*k+q_to_p],power)/Sigma[P*k+q_to_p];
)
funcdefine(
    _yesW_noU_noSigma_noPower,
    result[P*k+q_to_p] += W[K*n+k]*X[Q*n+q];
)
funcdefine(
    _yesW_noU_noSigma_yesPower,
    result[P*k+q_to_p] += W[K*n+k]*pow(X[Q*n+q],power);
)
funcdefine(
    _yesW_noU_yesSigma_noPower,
    result[P*k+q_to_p] += W[K*n+k]*X[Q*n+q]/Sigma[P*k+q_to_p];
)
funcdefine(
    _yesW_noU_yesSigma_yesPower,
    result[P*k+q_to_p] += W[K*n+k]*pow(X[Q*n+q],power)/Sigma[P*k+q_to_p];
)
funcdefine(
    _yesW_yesU_noSigma_noPower,
    result[P*k+q_to_p] += W[K*n+k]*(X[Q*n+q]-U[P*k+q_to_p]);
)
funcdefine(
    _yesW_yesU_noSigma_yesPower,
    result[P*k+q_to_p] += W[K*n+k]*pow(X[Q*n+q]-U[P*k+q_to_p],power);
)
funcdefine(
    _yesW_yesU_yesSigma_noPower,
    result[P*k+q_to_p] += W[K*n+k]*(X[Q*n+q]-U[P*k+q_to_p])/Sigma[P*k+q_to_p];
)
funcdefine(
    _yesW_yesU_yesSigma_yesPower,
    result[P*k+q_to_p] += W[K*n+k]*pow(X[Q*n+q]-U[P*k+q_to_p],power)/Sigma[P*k+q_to_p];
)

// interface that python will interact with
void polycomb(double *result, const double *X, const int64_t* mask, const int64_t* S, const double* W,
    const double* U, const double* Sigma, int64_t power, int64_t S_len, int64_t Q, int64_t N, int64_t P, int64_t K)
{
    int64_t SpecPower = power != 1;

    // choose the least general case for maximum performance
    if (!W && !U && !Sigma && !SpecPower)
        _noW_noU_noSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && !U && !Sigma && SpecPower)
        _noW_noU_noSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && !U && Sigma && !SpecPower)
        _noW_noU_yesSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && !U && Sigma && SpecPower)
        _noW_noU_yesSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && U && !Sigma && !SpecPower)
        _noW_yesU_noSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && U && !Sigma && SpecPower)
        _noW_yesU_noSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && U && Sigma && !SpecPower)
        _noW_yesU_yesSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (!W && U && Sigma && SpecPower)
        _noW_yesU_yesSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && !U && !Sigma && !SpecPower)
        _yesW_noU_noSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && !U && !Sigma && SpecPower)
        _yesW_noU_noSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && !U && Sigma && !SpecPower)
        _yesW_noU_yesSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && !U && Sigma && SpecPower)
        _yesW_noU_yesSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && U && !Sigma && !SpecPower)
        _yesW_yesU_noSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && U && !Sigma && SpecPower)
        _yesW_yesU_noSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else if (W && U && Sigma && !SpecPower)
        _yesW_yesU_yesSigma_noPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
    else
        _yesW_yesU_yesSigma_yesPower(result, X, mask, S, W, U, Sigma, power, S_len, Q, N, P, K);
}
