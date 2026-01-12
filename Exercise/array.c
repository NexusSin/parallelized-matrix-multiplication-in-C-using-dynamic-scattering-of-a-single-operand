#include "array.h"
#include <stdlib.h>
static unsigned int rand_seed = 12345;
void set_initilize_rand_seed(unsigned int seed) {
    rand_seed = seed;
    srand(seed);
}
double get_double_rand(void) {
    return (double)rand() / (double)RAND_MAX;
}
double* allocate_1d_double(int n) {
    return (double*)malloc(n * sizeof(double));
}
double* free_1d_double(double* arr) {
    if (arr != NULL) free(arr);
    return NULL;
}
void initialize_1d_double_rand(double* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = get_double_rand();
}
double* allocate_2d_double_blocked(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}
double* free_2d_double_blocked(double* arr) {
    if (arr != NULL) free(arr);
    return NULL;
}
void initialize_2d_double_blocked_rand(double* arr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) arr[i] = get_double_rand();
}
