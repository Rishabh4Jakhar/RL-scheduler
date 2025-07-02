#include "utils.h"

#include <stdlib.h>
#include <sys/time.h>
#ifdef OPENMP
#include<omp.h>
#endif

double get_time() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + time.tv_usec / 1000000.0;
#endif
}

double sum(double *a, unsigned long n) {
  double sum = 0.;
  for (int i = 0; i < n; ++i) {
    sum += a[i];
  }
  return sum;
}

double **matrix(size_t m, size_t n) {
  double *a = calloc(m * n, sizeof(double));
  double **b = malloc(m * sizeof(double *));

  for (int i = 0; i < m; ++i) {
    b[i] = a + i * n;
  }
  return b;
}

double ***tensor3d(size_t l, size_t m, size_t n) {
  double *a = calloc(l * m * n, sizeof(double));
  double **b = malloc(l * m * sizeof(double *));
  double ***c = malloc(l * sizeof(double **));

  for (int i = 0; i < l; ++i) {
    c[i] = b + i * m;
    for (int j = 0; j < m; ++j) {
      c[i][j] = a + j * n;
    }
  }
  return c;
}

void free_matrix(double **matrix) {
  free(matrix[0]);
  free(matrix);
}

void free_tensor3d(double ***tensor3d) {
  free(tensor3d[0][0]);
  free(tensor3d[0]);
  free(tensor3d);
}

double calculate_memory(Parameters *params) {
  return 1.e-9 * params->n_tallies * params->n_filter_bins * params->n_scores *
    params->n_nuclides * 3 * sizeof(double);
}
