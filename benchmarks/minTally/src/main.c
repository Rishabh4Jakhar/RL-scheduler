#ifdef OPENMP
#include<omp.h>
#endif
#include <stdio.h>
#include "io.h"
#include "random.h"
#include "simulation.h"
#include "tally.h"
#include "utils.h"

#if defined(HARMONIZER) || defined(LINUX)
	#include "hijack_main.h"
#endif

int main(int argc, char *argv[]) {
  // Load the problem input parameters
  printf("Initializing problem inputs...\n");
  Parameters *params = initialize_parameters(argc, argv);
  print_parameters(params);

  // Seed the RNG
  set_seed(params->seed);

  // Set the number of OpenMP threads
  #ifdef OPENMP
  printf("================= %d ================\n", params->n_threads);
  omp_set_num_threads(params->n_threads);
  #endif

  // Set up the tally structure
  printf("Initializing tallies...\n");
  Tally *tallies = initialize_tallies(params);

  // Simulation loop
  printf("Simulating tally events...\n");
  double start = get_time();
  unsigned long long verification = simulate(params, tallies);
  double end = get_time();

  // Print results
  print_border();
  printf("Verification checksum: %llu\n", verification);
  printf("Simulation time:       %e seconds\n", end - start);

  // Free memory
  free_tallies(tallies, params->n_tallies);
  free_parameters(params);

  return 0;
}
