#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

// Linear congruential random number generator: seed = (mult*seed + inc) % mod
double rn(uint64_t* seed);

// Sample a random integer in the range [a b)
int rni(int a, int b, uint64_t* seed);

// Set the master seed
void set_seed(uint64_t seed);

// Get the inital seed for the given ID
uint64_t init_seed(int64_t id);

// Algorithm to skip ahead n*rng.stride random numbers in O(log2(n)) operation,
// from 'The MCNP5 Random Number Generator', Forrest Brown, LA-UR-07K-7961.
void advance_seed(int64_t n, uint64_t* seed);

// Get the seed n skips ahead
uint64_t future_seed(uint64_t n, uint64_t seed);

// Sample a random number from a Gaussian distribution using Box-Muller
double sample_gaussian(double mu, double sigma, uint64_t *seed);

#endif
