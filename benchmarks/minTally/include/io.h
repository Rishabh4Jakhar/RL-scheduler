#ifndef IO_H
#define IO_H

#include "parameters.h"

// Read command line input
void read_input(int argc, char *argv[], Parameters *params);

// Print problem parameters
void print_parameters(Parameters *params);

// Print error message and exit
void print_error(char *message);

// Print border
void print_border(void);

// Prints comma separated integers - for ease of reading
void print_fancy_int(long a);

// Print Section titles in center of 80 char terminal
void print_center(const char *s, int width);

#endif
