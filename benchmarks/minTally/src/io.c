#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void read_input(int argc, char *argv[], Parameters *params) {
  // Collect raw input
  for (int i = 1; i < argc; ++i) {
    char *arg = argv[i];

    // Error message
    char message[1024];
    strcat(message, "Could not parse command line input '");
    strcat(message, arg);
    strcat(message, "'");

    if (strcmp(arg, "-r") == 0 || strcmp(arg, "--seed") == 0) {
      if (++i < argc) {
        params->seed = atoll(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--threads") == 0) {
      if (++i < argc) {
        params->n_threads = atoi(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--particles") == 0) {
      if (++i < argc) {
        params->n_particles = atoll(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--tallies") == 0) {
      if (++i < argc) {
        params->n_tallies = atoi(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--filter-bins") == 0) {
      if (++i < argc) {
        params->n_filter_bins = atoi(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--scores") == 0) {
      if (++i < argc) {
        params->n_scores = atoi(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--nuclides") == 0) {
      if (++i < argc) {
        params->n_nuclides = atoi(argv[i]);
      } else {
        print_error(message);
      }
    } else if (strcmp(arg, "-e") == 0 || strcmp(arg, "--events") == 0) {
      if (++i < argc) {
        params->n_events = atof(argv[i]);
      } else {
        print_error(message);
      }
    } else {
      print_error("Could not parse command line input");
    }
  }
  return;
}

void print_parameters(Parameters *params) {
  print_border();
  print_center("INPUT SUMMARY", 79);
  print_border();
  printf("RNG seed:                      %lu\n", (unsigned long)params->seed);
  printf("Number of OpenMP threads:      %d\n", params->n_threads);
  printf("Number of particles:           ");
  print_fancy_int(params->n_particles);
  printf("Number of tallies:             %d\n", params->n_tallies);
  printf("Number of filter bins:         %d\n", params->n_filter_bins);
  printf("Number of scores:              %d\n", params->n_scores);
  printf("Number of nuclides:            %d\n", params->n_nuclides);
  printf("Number of events per particle: %f\n", params->n_events);
  printf("Estimated memory usage:        %f GB\n", calculate_memory(params));
  print_border();
}

void print_error(char *message) {
  printf("ERROR: %s\n", message);
  exit(1);
}

void print_border(void) {
  printf("=========================================="
         "======================================\n");
}

void print_fancy_int(long a) {
  if (a < 1000) {
    printf("%ld\n", a);
  } else if (a >= 1000 && a < 1000000) {
    printf("%ld,%03ld\n", a / 1000, a % 1000);
  } else if (a >= 1000000 && a < 1000000000) {
    printf("%ld,%03ld,%03ld\n", a / 1000000, (a % 1000000) / 1000, a % 1000);
  } else if (a >= 1000000000) {
    printf("%ld,%03ld,%03ld,%03ld\n", a / 1000000000,
           (a % 1000000000) / 1000000, (a % 1000000) / 1000, a % 1000);
  } else {
    printf("%ld\n", a);
  }
}

void print_center(const char *s, int width) {
  for (int i = 0; i <= (width - strlen(s)) / 2; ++i) {
    fputs(" ", stdout);
  }
  fputs(s, stdout);
  fputs("\n", stdout);
}
