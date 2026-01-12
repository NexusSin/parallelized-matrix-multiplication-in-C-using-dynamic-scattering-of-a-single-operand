#ifndef ARRAY_H
#define ARRAY_H
#include <stdlib.h>
void set_initilize_rand_seed(unsigned int seed);
double get_double_rand(void);
double* allocate_1d_double(int n);
double* free_1d_double(double* arr);
void initialize_1d_double_rand(double* arr, int n);
double* allocate_2d_double_blocked(int rows, int cols);
double* free_2d_double_blocked(double* arr);
void initialize_2d_double_blocked_rand(double* arr, int rows, int cols);
#endif
