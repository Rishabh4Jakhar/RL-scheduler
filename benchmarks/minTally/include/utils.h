#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include "parameters.h"

// Get the current time in seconds
double get_time();

// Compute the sum of array elements
double sum(double* a, unsigned long n);

// 2D matrix of doubles contiguous in memory initialized to zero
double** matrix(size_t m, size_t n);

// 3D tensor of doubles contiguous in memory initialized to zero
double*** tensor3d(size_t l, size_t m, size_t n);

// Free matrix memory
void free_matrix(double** matrix);

// Free 3D tensor memory
void free_tensor3d(double*** tensor3d);

// Calculate the memory footprint in GB of the tally data
double calculate_memory(Parameters* params);

#endif
