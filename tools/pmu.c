#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "pmu.h"

static int **fds = NULL;                                                          // File descriptors for core-event pairs
static int num_cores = 40;                                                        // Number of cores
static int num_physical_cores = 20;
static int num_sockets = 2;                                                       // Number of sockets

static uint64_t *prev_core_values = NULL;                                         // Stores previous counter values per core
static uint64_t *curr_core_values = NULL;                                         // Stores current counter values per core
static uint64_t *diff_core_values = NULL; 

static uint64_t *prev_socket_values = NULL;                                       // Stores previous counter values per socket
static uint64_t *curr_socket_values = NULL;                                       // Stores current counter values per socket
static uint64_t *diff_socket_values = NULL;
double Joule_conversion_unit = 0;

const char* EVENTS[5]= {  "INSTRUCTION_RETIRED",
                          "PERF_COUNT_HW_CPU_CYCLES" /*TSC counter*/,
                          "RAPL_ENERGY_PKG",
                          "MEM_LOAD_UOPS_RETIRED.L3_HIT",
                          "MEM_LOAD_UOPS_RETIRED.L3_MISS"
                       };

int get_cores_in_socket(int socket_id) {
  return num_physical_cores / num_sockets;  
}

int get_core_from_socket(int socket_id, int index) {
  return socket_id * get_cores_in_socket(socket_id) + index;
}

static int open_perf_event(struct perf_event_attr *pe, int core) {
  int fd = perf_event_open(pe, -1, core, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
    exit(1);
  }
  return fd;
}

uint64_t read_event(int core_id, int event_index) {
  if (core_id < 0 || core_id >= num_cores || event_index < 0 || event_index >= NUM_EVENTS) {
    fprintf(stderr, "Invalid core or event index\n");
    return 0;
  }

  int fd = fds[core_id][event_index];
  if (fd == -1) {
    fprintf(stderr, "Invalid file descriptor for core %d, event %s\n", core_id, EVENTS[event_index]);
    return 0;
  }

  uint64_t value = 0;
  if (read(fd, &value, sizeof(value)) != sizeof(value)) {
    perror("read");
    return 0;
  }

  return value;
}

void libpfm_init() {

  /***** Initialize the file descriptor, previous and current value arrays ****/
  fds = (int**) malloc(num_cores * sizeof(int*));

  prev_core_values = (uint64_t*) malloc(num_cores * NUM_EVENTS * sizeof(uint64_t));
  curr_core_values = (uint64_t*) malloc(num_cores * NUM_EVENTS * sizeof(uint64_t));
  diff_core_values = (uint64_t*) malloc(num_cores * NUM_EVENTS * sizeof(uint64_t));

  prev_socket_values = (uint64_t*) malloc(num_sockets * NUM_EVENTS * sizeof(uint64_t));
  curr_socket_values = (uint64_t*) malloc(num_sockets * NUM_EVENTS * sizeof(uint64_t));
  diff_socket_values = (uint64_t*) malloc(num_sockets * NUM_EVENTS * sizeof(uint64_t));

  memset(prev_core_values, 0, num_cores * NUM_EVENTS * sizeof(uint64_t));
  memset(curr_core_values, 0, num_cores * NUM_EVENTS * sizeof(uint64_t));
  memset(diff_core_values, 0, num_cores * NUM_EVENTS * sizeof(uint64_t));

  memset(prev_socket_values, 0, num_sockets * NUM_EVENTS * sizeof(uint64_t));
  memset(curr_socket_values, 0, num_sockets * NUM_EVENTS * sizeof(uint64_t));
  memset(diff_socket_values, 0, num_sockets * NUM_EVENTS * sizeof(uint64_t));

  if (!fds || !prev_core_values || !curr_core_values) {
    perror("malloc/calloc");
    exit(1);
  }

  for (int i = 0; i < num_cores * NUM_EVENTS; i++) {                                                 // Initially set all values to 0
    prev_core_values[i] = 0;
    curr_core_values[i] = 0;
    diff_core_values[i] = 0;
  }

  for (int i = 0; i < num_sockets * NUM_EVENTS; i++) {                                                // Initially set all values to 0
    prev_socket_values[i] = 0;
    curr_socket_values[i] = 0;
    diff_socket_values[i] = 0;
  }

  /**** Initialize libpfm ***/
  if (pfm_initialize() != PFM_SUCCESS) {
    fprintf(stderr, "Failed to initialize libpfm\n");
    exit(1);
  }

  for (int core = 0; core < num_cores; core++) {
    fds[core] = (int*) malloc(NUM_EVENTS * sizeof(int));
    if (!fds[core]) {
      perror("malloc");
      exit(1);
    }

    /**** Set up each counter ***/
    struct perf_event_attr pe;
    for (int i = 0; i < NUM_EVENTS; i++) {                                                           
      memset(&pe, 0, sizeof(struct perf_event_attr));
      pe.size = sizeof(struct perf_event_attr);

      int ret = pfm_get_perf_event_encoding(EVENTS[i], PFM_PLM3, &pe, NULL, NULL);                  // Encode the event for perf_event
      if (ret != PFM_SUCCESS) {
        fprintf(stderr, "Failed to encode event %s (%s)\n", EVENTS[i], pfm_strerror(ret));
        fds[core][i] = -1;
        continue;
      }

      fds[core][i] = open_perf_event(&pe, core);                                                    // Open file descriptor for counter per core
    }
  }

  Joule_conversion_unit = pow(2, -32);
}

void libpfm_finalize() {
  free(curr_core_values);
  free(prev_core_values);
  free(curr_socket_values);
  free(prev_socket_values);
  free(diff_core_values);
  free(diff_socket_values);

  for (int core = 0; core < num_cores; core++) {
    if (fds[core]) {  // Ensure fds[core] is allocated
      for (int i = 0; i < NUM_EVENTS; i++) {
        if (fds[core][i] >= 0) {
          close(fds[core][i]);  // Close the file descriptor
        }
      }
      free(fds[core]);  // Free the allocated memory for this core
    }
  }
  free(fds);  // Free the top-level array

}

void accumulate_pmcs() {
  for(int event_index=0; event_index < NUM_EVENTS; event_index++) {
    if(strstr(EVENTS[event_index], "RAPL") || strstr(EVENTS[event_index], "UNC")) {
      continue;
    }

    for(int core_id=0; core_id < num_cores; core_id++) {
      int index = core_id * NUM_EVENTS + event_index;
      curr_core_values[index] = read_event(core_id, event_index);

      double diff = (double)(curr_core_values[index] - prev_core_values[index]);
      diff_core_values[index] = diff;
      // assert(diff >=0, "");                                                                             // Add assertion here.

      prev_core_values[index] = curr_core_values[index];  
    }
  }
}

void accumulate_cbox() {
  uint64_t total_value = 0;

  for(int event_index=0; event_index<NUM_EVENTS; event_index++) {
    if(strstr(EVENTS[event_index], "RAPL") || strstr(EVENTS[event_index], "UNC")) {
      for (int socket_id=0; socket_id < num_sockets; socket_id++) {
        int index = socket_id * NUM_EVENTS + event_index;
        int num_cores_in_socket = get_cores_in_socket(socket_id);

        for (int i = 0; i < num_cores_in_socket; i++) {
          int core_id = get_core_from_socket(socket_id, i);
          double temp = read_event(core_id, event_index);
          total_value += temp;
        }

        curr_socket_values[index] = total_value;

        double diff = (double)(curr_socket_values[index] - prev_socket_values[index]);
        // assert(diff >=0, "");                                                                             // Add assertion here.
        prev_socket_values[index] = total_value;
      }
    }
  }
}

void accumulate_energy() {
  int num_cores_in_socket = get_cores_in_socket(0);

  for(int event_index=0; event_index<NUM_EVENTS; event_index++) {
    if(strstr(EVENTS[event_index], "RAPL")) {
      for (int socket_id=0; socket_id < num_sockets; socket_id++) {

        if (socket_id >= num_sockets || event_index >= NUM_EVENTS) {
          fprintf(stderr, "Invalid socket or event index\n");
          return ;
        }

        uint64_t total_value = 0;
        for (int i = 0; i < 1; i++) {
          int core_id = get_core_from_socket(socket_id, i);
          total_value += read_event(core_id, event_index);
        }

        // Save current value and calculate difference
        int index = socket_id * NUM_EVENTS + event_index;
        curr_socket_values[index] = total_value;

        double diff = (double)(curr_socket_values[index] - prev_socket_values[index]);
        
        // assert(diff >=0, "");                                                                 // Add assertion here.
    
        prev_socket_values[index] = total_value;

        if(strstr(EVENTS[event_index], "RAPL_ENERGY_PKG"))
          diff_socket_values[index] = diff * Joule_conversion_unit;
        else
          diff_socket_values[index] = diff;
      }
    }
  }
}

double getEventOnCore(int core_id, int event_index) {
  if (core_id < 0 || core_id >= num_cores || event_index < 0 || event_index >= NUM_EVENTS) {
    // fprintf(stderr, "Invalid core or event index\n");
    return -1;
  }

  // Save current value and calculate difference
  int index = core_id * NUM_EVENTS + event_index;
  return diff_core_values[index];
}

double getEventOnSocket(int socket_id, int event_index) {
  int index = socket_id * NUM_EVENTS + event_index;
  if(strstr(EVENTS[event_index], "RAPL_ENERGY_PKG")) {
    return diff_socket_values[index];
  }
  
  if (socket_id >= num_sockets || event_index >= NUM_EVENTS) {
    // fprintf(stderr, "Invalid socket or event index\n");
    return -1;
  }

  return diff_socket_values[index];
}
