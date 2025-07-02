#include "tally.h"

#include "random.h"
#include "utils.h"

Tally *initialize_tallies(Parameters *params) {
  // Get the RNG seed
  uint64_t seed = init_seed(0);

  Tally *tallies = malloc(params->n_tallies * sizeof(Tally));

  for (int i = 0; i < params->n_tallies; ++i) {
    tallies[i].n_scores = params->n_nuclides * params->n_scores;
    tallies[i].n_filter_bins = params->n_filter_bins;
    tallies[i].filter_bins = malloc(params->n_filter_bins * sizeof(int));

    for (int j = 0; j < params->n_filter_bins; ++j) {
        tallies[i].filter_bins[i] = rni(0, params->n_coord, &seed);
    }

    // Allocate space for each filter bin and score combination and the three
    // types of tally results
    tallies[i].results =
      tensor3d(tallies[i].n_filter_bins, tallies[i].n_scores, 3);
  }
  return tallies;
}

void free_tallies(Tally *tallies, int n) {
  for (int i = 0; i < n; ++i) {
    free(tallies[i].filter_bins);
    free_tensor3d(tallies[i].results);
  }
  free(tallies);
}
