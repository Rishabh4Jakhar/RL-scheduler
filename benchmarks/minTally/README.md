# minTally

minTally is a mini-app that abstracts the key components of the tallying system from a full Monte Carlo neutron transport application. Specifically, it represents the tally scoring region of the [OpenMC](https://github.com/openmc-dev/openmc) code as applied to a realistic Monte Carlo reactor core problem that requires extremely large tally data structures. It completely abstracts out the complex tracking physics of the full application while seeking to retain the key perforamance features of the tally system, making it a suitable stand-in for the full code for testing new hardware features and programming models.

## Compilation

To compile minTally with default settings, use the following command:

```bash
make
```

Compiler settings can be modified in the Makefile, and the following options can also be set:

| Option | Description| Default |
|-------------|------------|---------------|
| OPENMP | Enables shared-memory parallelism using the OpenMP API | ON |
| OPTIMIZE | Enables high optimization with the -O3 flag | ON|
| DEBUG | Enables debugging with the -g flag | OFF |
| PROFILE | Enables profiling with the -pg flag | OFF |

## Running

To run minTally with default settings, use the following command:
```bash
./bin/minTally
```

The settings can be changed using the following command line flags:

| Option | Description| Default |
|-------------|------------|---------------|
| -c, --threads | Number of OpenMP threads | System default |
| -r, --seed | RNG seed | 12345 |
| -p, --particles | Number of particles | 100000 |
| -t, --tallies | Number of tallies | 180 |
| -f, --filter-bins | Number of tally filter bins | 50952 |
| -s, --scores | Number of tally scores | 6 |
| -n, --nuclides | Size of the nuclide inventory | 6 |
| -e, --events | Mean number of tally events per particle | 5.4 |

- [**-p, --particles**]
Sets the number of particle histories to simulate. Increasing the number of particles will extend the runtime without affecting the memory requirements.

- [**-t, --tallies**]
Sets the number of tallies. Tallies are the results of a simulation -- they are the estimated physical quantities such as flux or reaction rate. Increasing the number of talliese here would be analogous to increasing the number of geometric regions in a fuel pin in a MC reactor benchmark such as BEAVRS. The memory requirements will be proportional to the number of tallies specified.

- [**-f, --filter-bins**]
Sets the number of filter bins defined for each tally. A filter limits what events can score to a given tally; for example, a filter criterion might be that a particle has a collision within a particular cell or that the particle's energy is within a defined range. The default number of filter bins is chosen to correspond to the BEAVRS benchmark.

- [**-s, --scores**]
Sets the number of scoring functions. Scores are the actual physical quantities to be tallied (e.g. flux, reaction rate). By default, the number of scores is set to 6 and represents microscopic reaction rates. If an event satisfies a filter criterion, the tally result will be incremented for each score and for every nuclide in the problem. The total size of the results array for each tally is n_filter_bins x n_scores x n_nuclides x 3 (for the three types of tally results: value, sum, and sum squared). Therefore, increasing the number of filter bins, scores, or nuclides will increase the memory footprint.

- [**-e, --events**]
Sets the average number of tally events per particle. This represents, e.g., the number of particle collisions or tracks in fuel. The default value of 5.4 was estimated from the BEAVRS benchmark and represents the mean number of collisions in fuel per particle. Increasing the number of events will increase the runtime but will not affect the memory requirements.

## Verification

A verification checksum is generated during the simulation and displayed with the results. This can be used as a reference to check the correctness of other versions or configurations of the code. Varying the number of OpenMP threads should not alter the results, but changing the other runtime options will result in a different verification checksum.
