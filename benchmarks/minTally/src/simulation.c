#include "simulation.h"

#include <math.h>
#ifdef OPENMP
#include<omp.h>
#endif
#include "random.h"

unsigned long long simulate(Parameters *params, Tally *tallies) {
  unsigned long long verification = 0;

  // Loop over particles
  #pragma omp parallel for schedule(static) reduction(+:verification)
  for (unsigned long i_p = 0; i_p < params->n_particles; ++i_p) {
    // Get the RNG seed
    uint64_t seed = init_seed(i_p + 1);

    const double weight = 1.;
    double distance = rn(&seed);
    double flux = weight * distance;

    // Sample a number of tallying events (e.g. number of collisions or tracks)
    // for this particle
    int n_event = sample_n_events(params, &seed);

    // Loop over tallying events
    for (int i_e = 0; i_e < n_event; ++i_e) {
      // Sample a cell containing fuel material to score to by randomly
      // selecting a filter bin and tally index
      int filter_index = rni(0, params->n_filter_bins, &seed);
      int tally_index = rni(0, params->n_tallies, &seed);

      // Loop over tallies
      for (int i_t = 0; i_t < params->n_tallies; ++i_t) {
        // Search the levels of the nested geometry for the sampled cell
        int i_c;
        for (i_c = 0; i_c < params->n_coord; ++i_c) {
          if (i_c == tallies[i_t].filter_bins[filter_index]) {
            break;
          }
        }

        // If this cell does not score to this tally, continue to the next
        // tally/filter; otherwise, score to this tally
        if (tally_index != i_t) {
          continue;
        }

        // Loop over nuclides. We are only scoring to tallies in fuel, so we
        // will always loop over all the nuclides, which represent the
        // nuclides in the fuel material.
        for (int i_n = 0; i_n < params->n_nuclides; ++i_n) {
          // Get the atomic density
          double atom_density = rn(&seed);

          // Get the microscopic cross section
          double micro_xs = rn(&seed);

          // Loop over scores
          for (int i_s = 0; i_s < params->n_scores; ++i_s) {
            // Determine the score index in the tally array
            int score_index = i_n * params->n_scores + i_s;

            // For tracklength tallies, the score is computed as
            // micro_xs.total * atom_density * flux, where flux =
            // particle_weight * distance
            double score = micro_xs * atom_density * flux;

            // Accumulate tally
            #pragma omp atomic
            tallies[i_t].results[filter_index][score_index][VALUE] += score;

            // Update verification checksum
            unsigned long long *ptr = (unsigned long long*)&score;
            verification += *ptr;
          }
        }
        verification += i_c;
      }
    }
  }
  return verification;
}

int sample_n_events(Parameters *params, uint64_t *seed) {
  // Sample the number of events from a normal distribution with mean and
  // standard deviation determined by the tally estimator and the number of
  // radial regions in fuel
  double z;
  do {
    z = sample_gaussian(params->n_events, 0.5 * params->n_events, seed);
  } while (z < 0);
  int n = round(z);
  return n;
}
