#include "multiply.h"
#include <math.h>
#include <stdlib.h>
#define MAT(ptr, i, j, ld) ((ptr)[(i) + (j) * (ld)])
void multiply_matrices(int const m, int const k, int const n,
                      double const* const A, double const* const B, 
                      double* const C) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) MAT(C, i, j, m) = 0.0;
    }
    for (int j = 0; j < n; j++) {
        for (int l = 0; l < k; l++) {
            double const b_lj = MAT(B, l, j, k);
            for (int i = 0; i < m; i++) {
                MAT(C, i, j, m) += MAT(A, i, l, m) * b_lj;
            }
        }
    }
}
bool is_product(int const m, int const k, int const n,
               double const* const A, double const* const B, 
               double const* const C, double const epsilon) {
    double* C_test = (double*)malloc(m * n * sizeof(double));
    multiply_matrices(m, k, n, A, B, C_test);
    double max_error = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double diff = fabs(MAT(C, i, j, m) - MAT(C_test, i, j, m));
            if (diff > max_error) max_error = diff;
        }
    }
    free(C_test);
    return max_error < epsilon;
}
