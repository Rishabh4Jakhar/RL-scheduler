#ifndef TALLY_H
#define TALLY_H

#include <stdlib.h>
#include "parameters.h"

// Tally results
#define VALUE 0
#define SUM 1
#define SUM_SQ 2

typedef struct {
  // Number of physical quantities to be tallied, e.g. reaction rates
  int n_scores;

  // Filter combinations, e.g. cell and energy
  int n_filter_bins;
  int *filter_bins;

  // Tally results for each bin. The first dimension is the combination of
  // filters (e.g. specific cell). The second dimension is the score (e.g.
  // flux, reaction rate). The third dimension is for the three tally result
  // types: the tally value for this cycle, the accumulated sum over all
  // cycles, and the sum squared over all cycles.
  double ***results;
} Tally;

// Allocate memory for tallies
Tally *initialize_tallies(Parameters *params);

// Free memory
void free_tallies(Tally *tallies, int n);

#endif
