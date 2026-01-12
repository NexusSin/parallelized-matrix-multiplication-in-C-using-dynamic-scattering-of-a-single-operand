#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "array.h"
#include "multiply.h"

static unsigned int const seed = 1234;
static int const dimensions[] = {128*1, 128*2, 128*4, 128*8};
static int const n_dimensions = sizeof(dimensions)/sizeof(int);
static double const epsilon = 1e-10;

typedef void (*GEMM)(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C
);

static void populate_compatible_random_matrix_pairs(
    int const m, int const k, int const n,
    int const seed,
    double* const A, double* const B)
{
    set_initilize_rand_seed(seed);
    initialize_2d_double_blocked_rand(A, m, k);
    initialize_2d_double_blocked_rand(B, k, n);
}

static void initialize_problem_matrices(
    int const m, int const k, int const n,
    double** const A, double** const B, double** const C)
{
    *A = allocate_2d_double_blocked(m, k);
    *B = allocate_2d_double_blocked(k, n);
    *C = allocate_2d_double_blocked(m, n);
}

static void destroy_problem_matrices(double** const A, double** const B, double** const C)
{
    *A = free_2d_double_blocked(*A);
    *B = free_2d_double_blocked(*B);
    *C = free_2d_double_blocked(*C);
}

static bool test_multiply(int const m, int const k, int const n, GEMM gemm, double const epsilon, unsigned int const seed, double* const duration)
{
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);
    
    clock_t const start = clock();
    gemm(m, k, n, A, B, C);
    clock_t const end = clock();
    
    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);
    destroy_problem_matrices(&A, &B, &C);
    *duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    return result_is_correct;
}

// ============================================================================
// SOLUTION: DGEMM implementations
// ============================================================================

// Macro for column-major indexing
#define MAT(ptr, i, j, ld) ((ptr)[(i) + (j) * (ld)])

// Column-major DGEMM (j->l->i loop order, cache-friendly)
void DGEMM(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    // Initialize C to zero
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            MAT(C, i, j, m) = 0.0;
        }
    }
    
    // C <- A * B
    // Outer loop over columns of C (j)
    // Middle loop over columns of A / rows of B (l)
    // Inner loop over rows of C (i)
    // This ensures A is accessed column-wise (sequential in memory)
    for (int j = 0; j < n; j++) {
        for (int l = 0; l < k; l++) {
            double const b_lj = MAT(B, l, j, k);
            for (int i = 0; i < m; i++) {
                MAT(C, i, j, m) += MAT(A, i, l, m) * b_lj;
            }
        }
    }
}

// Row-wise DGEMM (j->i->l loop order, accesses A row-wise)
void rowwise_DGEMM(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    // Initialize C to zero
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            MAT(C, i, j, m) = 0.0;
        }
    }
    
    // C <- A * B
    // Outer loop over columns of C (j)
    // Middle loop over rows of C (i)
    // Inner loop performs dot product of row i of A with column j of B
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += MAT(A, i, l, m) * MAT(B, l, j, k);
            }
            MAT(C, i, j, m) = sum;
        }
    }
}

// ============================================================================

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n)
{
    int const max_dim = n_dimensions;
    static int dim = 0;
    if (dim >= max_dim) {
        return false;
    }
    *m = dimensions[dim];
    *k = dimensions[dim];
    *n = dimensions[dim];
    dim++;
    return true;
}

int main(int argc, char* argv[])
{
    bool all_test_pass = true;
    int m = 0;
    int k = 0;
    int n = 0;
    
    while (generate_square_matrix_dimension(&m, &k, &n)) {
        double columnwise_duration = 0.0;
        bool const test_DGEMM_pass = test_multiply(m, k, n, DGEMM, epsilon, seed, &columnwise_duration);
        if (!test_DGEMM_pass) {
            printf("DGEMM failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
        
        double rowwise_duration = 0.0;
        bool const test_rowwise_pass = test_multiply(m, k, n, rowwise_DGEMM, epsilon, seed, &rowwise_duration);
        if (!test_rowwise_pass) {
            printf("rowwise_DGEMM failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
        
        printf("Duration for m=%d, k=%d, n=%d: DGEMM=%lf, rowwise_DGEMM=%lf\n", 
               m, k, n, columnwise_duration, rowwise_duration);
    }
    
    if (!all_test_pass) {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
