#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdint.h>

typedef struct {
    uint64_t seed;
    int n_threads;
    unsigned long n_particles;
    int n_tallies;
    int n_filter_bins;
    int n_scores;
    int n_nuclides;
    int n_coord;
    double n_events;
} Parameters;

// Set the default parameters and read CLI
Parameters *initialize_parameters(int argc, char *argv[]);

// Free memory
void free_parameters(Parameters *params);

#endif
