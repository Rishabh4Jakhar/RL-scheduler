#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <sys/syscall.h>

void timer_func(double *timer){
  struct timespec currentTime;
  clock_gettime (CLOCK_MONOTONIC, &currentTime);
  *timer = (currentTime.tv_sec + (currentTime.tv_nsec * 10e-10));
}

int user_main(int argc, char** argv);

int main(int argc, char **argv) {
  double start_def, end_def;
  /*
   * This pragma is to pin the OpenMP threads
   * according to the affinity set in launcher
   */
  #pragma omp parallel
    usleep(10);

  fprintf(stdout,"\n-----\nmkdir timedrun fake\n\n");
  timer_func(&start_def);

  int x = user_main(argc, argv);

  timer_func(&end_def);
  fprintf(stdout,"\n============================ Tabulate Statistics ============================\n");
  fprintf(stdout,"%s\n","TIME");
  fprintf(stdout,"%f\n",(end_def-start_def)*1000); // return total time
  fprintf(stdout,"=============================================================================\n");
  fprintf(stdout,"===== Test PASSED in 0.0 msec =====\n");
  fflush(stdout);

  return x;
}

#define main user_main

