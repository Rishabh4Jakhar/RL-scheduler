#include "parameters.h"

#ifdef OPENMP
#include<omp.h>
#endif
#include <stdlib.h>
#include "io.h"

Parameters *initialize_parameters(int argc, char *argv[]) {
  Parameters *params = malloc(sizeof(Parameters));

  // Assign default parameters
  params->seed = 12345;
  // #ifdef OPENMP
  // params->n_threads = omp_get_num_procs();
  // #else
  params->n_threads = atoi(getenv("OMP_NUM_THREADS"));
  // #endif
  params->n_particles = 1000000000;
  params->n_tallies = 180;
  params->n_filter_bins = 1000;
  params->n_scores = 1;
  params->n_nuclides = 1;
  params->n_coord = 10;

  // The number of tally events per particle is the average number of
  // collisions or track in fuel. The default numbers were estimated from the
  // BEAVRS, ExaSMR, and Monte Carlo performance benchmarks and are approxmatly
  // the same for each benchmark. The mean number of collisions was 5.4 with
  // standard deviation 3.2, and the mean number of tracks was 20.7 with
  // standard deviation 10.5. The mean number of track in fuel scales linearly
  // with the number of radial regions.
  params->n_events = 4;

  // Read command line arguments
  // read_input(argc, argv, params);

  return params;
}

void free_parameters(Parameters *params) { free(params); }
