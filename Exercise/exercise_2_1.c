#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
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

static bool test_multiply(int const m, int const k, int const n, GEMM gemm, double const epsilon, unsigned int const seed)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    
    if (rank == 0) {
        populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);
    }
    
    MPI_Bcast(A, m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    gemm(m, k, n, A, B, C);
    
    bool result_is_correct = true;
    if (rank == 0) {
        result_is_correct = is_product(m, k, n, A, B, C, epsilon);
        printf("Multiplication %s for: m=%d, k=%d, n=%d\n", 
               result_is_correct ? "passed" : "failed", m, k, n);
    }
    
    MPI_Bcast(&result_is_correct, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    destroy_problem_matrices(&A, &B, &C);
    return result_is_correct;
}

#define MAT(ptr, i, j, ld) ((ptr)[(i) + (j) * (ld)])

void parallel_gemm(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int const n_local = n / size;
    
    double* B_local = allocate_2d_double_blocked(k, n_local);
    double* C_local = allocate_2d_double_blocked(m, n_local);
    
    if (rank == 0) {
        for (int j = 0; j < n_local; j++) {
            for (int i = 0; i < k; i++) {
                MAT(B_local, i, j, k) = MAT(B, i, j, k);
            }
        }
        
        for (int p = 1; p < size; p++) {
            double* B_send = allocate_2d_double_blocked(k, n_local);
            for (int j = 0; j < n_local; j++) {
                for (int i = 0; i < k; i++) {
                    MAT(B_send, i, j, k) = MAT(B, i, p * n_local + j, k);
                }
            }
            MPI_Send(B_send, k * n_local, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            B_send = free_2d_double_blocked(B_send);
        }
    } else {
        MPI_Recv(B_local, k * n_local, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    for (int j = 0; j < n_local; j++) {
        for (int i = 0; i < m; i++) {
            MAT(C_local, i, j, m) = 0.0;
        }
    }
    
    for (int j = 0; j < n_local; j++) {
        for (int l = 0; l < k; l++) {
            double const b_lj = MAT(B_local, l, j, k);
            for (int i = 0; i < m; i++) {
                MAT(C_local, i, j, m) += MAT(A, i, l, m) * b_lj;
            }
        }
    }
    
    if (rank == 0) {
        for (int j = 0; j < n_local; j++) {
            for (int i = 0; i < m; i++) {
                MAT(C, i, j, m) = MAT(C_local, i, j, m);
            }
        }
        
        for (int p = 1; p < size; p++) {
            double* C_recv = allocate_2d_double_blocked(m, n_local);
            MPI_Recv(C_recv, m * n_local, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < n_local; j++) {
                for (int i = 0; i < m; i++) {
                    MAT(C, i, p * n_local + j, m) = MAT(C_recv, i, j, m);
                }
            }
            C_recv = free_2d_double_blocked(C_recv);
        }
    } else {
        MPI_Send(C_local, m * n_local, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    B_local = free_2d_double_blocked(B_local);
    C_local = free_2d_double_blocked(C_local);
}

GEMM const tested_gemm = &parallel_gemm;

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
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    bool all_test_pass = true;
    int m = 0;
    int k = 0;
    int n = 0;
    
    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool const test_pass = test_multiply(m, k, n, tested_gemm, epsilon, seed);
        if (!test_pass) {
            all_test_pass = false;
        }
    }
    
    MPI_Finalize();
    
    if (!all_test_pass) {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
