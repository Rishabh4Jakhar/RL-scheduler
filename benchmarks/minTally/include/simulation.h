#ifndef SIMULATION_H
#define SIMULATION_H

#include "parameters.h"
#include "tally.h"

// Loop over particles and tally
unsigned long long simulate(Parameters *params, Tally *tallies);

// Sample the number of events each particle undergoes in its lifetime
int sample_n_events(Parameters *params, uint64_t *seed);

#endif
