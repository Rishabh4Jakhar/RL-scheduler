#################################################### IMPORT LIBRARIES ####################################################
import ctypes
import ctypes.util
import subprocess
import time
import os
import signal
import posix_ipc
import mmap
import struct
import shlex
import threading
import psutil
import math
import time
import getpass
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from pyDOE import lhs
import numpy as np

#################################################### PATHS & VARIABLES ####################################################
username = getpass.getuser()
current_dir = "/home/" + username + "/profiler"
LIB_PATH = current_dir + "/tools/libpmu.so"
LIB_HIJACK = current_dir + "/tools/libhijack.so"
SHM_NAME = "__tid_shared_memory"

benchmark_configs = []
Benchnames = []

core_assignments = []

MAX_BENCHMARKS = 1
CORES_PER_APP = 10
THREADS_PER_CORE = 2
THREADS_PER_APP = 20
TOTAL_PHYSICAL_CORES = 20
TID_SIZE = ctypes.sizeof(ctypes.c_int)
TOTAL_TIDS = MAX_BENCHMARKS * THREADS_PER_APP

SLEEP_INTERVAL = 0.05

INSTRUCTIONS = []
CYCLES = []
IPC_sample = [0.0] * MAX_BENCHMARKS
TIME = [0.0] * MAX_BENCHMARKS
IPC_FINAL = [0.0] * MAX_BENCHMARKS
ENERGY = 0

#################################################### LOAD SHARED LIBRARIES ####################################################
profiler = ctypes.CDLL(LIB_PATH)
hijack = ctypes.CDLL(LIB_HIJACK)

profiler.getEventOnCore.argtypes = [ctypes.c_int, ctypes.c_int]
profiler.getEventOnCore.restype = ctypes.c_ulonglong

# Define function prototypes
profiler.libpfm_init.argtypes = []
profiler.libpfm_init.restype = None

profiler.libpfm_finalize.argtypes = []
profiler.libpfm_finalize.restype = None

profiler.accumulate_energy.argtypes = []
profiler.accumulate_energy.restype = None

profiler.accumulate_pmcs.argtypes = []
profiler.accumulate_pmcs.restype = None

profiler.accumulate_cbox.argtypes = []
profiler.accumulate_cbox.restype = None

profiler.getEventOnCore.argtypes = [ctypes.c_int, ctypes.c_int]  
profiler.getEventOnCore.restype = ctypes.c_double  

profiler.getEventOnSocket.argtypes = [ctypes.c_int, ctypes.c_int]  
profiler.getEventOnSocket.restype = ctypes.c_double  

class SharedData(ctypes.Structure):
  _fields_ = [("thread_ids", ctypes.c_int * TOTAL_TIDS)]

LD_PRELOAD = current_dir + "/tools/libhijack.so:" + current_dir + "/tools/libpmu.so"

#################################################### DAEMON FUNCTIONS ####################################################

def get_benchmark_configs():
  # Get the environment variables
  benchmark_names = os.getenv('BENCHMARK_NAMES', '').strip()
  benchmark_binaries = os.getenv('BENCHMARK_BINARIES', '').strip()
  ranges = os.getenv('RANGES', '').strip()

  if not benchmark_names or not benchmark_binaries or not ranges:
    print("Error: BENCHMARK_NAMES, BENCHMARK_BINARIES, or RANGES not set.")
    return 

  # Split the values into lists
  benchmark_names_list = benchmark_names.split(',') if benchmark_names else []
  benchmark_binaries_list = benchmark_binaries.split(',') if benchmark_binaries else []
  ranges_list = ranges.split()

  print(benchmark_names_list)

  # Ensure that the number of benchmarks, binaries, and ranges match
  if len(benchmark_names_list) != len(benchmark_binaries_list) or len(benchmark_names_list) != len(ranges_list):
    print("Error: The number of benchmark names, binaries, and ranges do not match.")
    return 

  # Append the new data to the existing lists
  benchmark_configs.extend((binary, range_) for binary, range_ in zip(benchmark_binaries_list, ranges_list))
  Benchnames.extend(benchmark_names_list)

  print(benchmark_configs)

def assign_cores():
  for _, core_str in benchmark_configs:
    core_list = []
    for core in core_str.split(","):
      core_id = int(core.strip())
      core_list.append(core_id)
      core_list.append(core_id + TOTAL_PHYSICAL_CORES)

    core_assignments.append(core_list)

  print(core_assignments)

######### FUNCTION 0: PIN THREAD TO CORE ###################################################################
def pin_thread_to_core(tid, core_id):
  try:
    # Get the process with the thread ID
    process = psutil.Process(tid)
    
    # Pin the thread to the specific core
    process.cpu_affinity([core_id])
    
    # Print confirmation
    print(f"Thread {tid} is pinned to CPU core {core_id}")
  except psutil.NoSuchProcess:
    print(f"Thread {tid} does not exist.")
  except Exception as e:
    print(f"Error pinning thread {tid} to core {core_id}: {e}")

def pin_thread_to_cores():
  tids = get_tids_for_benchmark(SHM_NAME, TOTAL_TIDS)
  for bench_id in range(MAX_BENCHMARKS):
    start_tid_bench=bench_id*THREADS_PER_APP 
    end_tid_bench=start_tid_bench+(THREADS_PER_APP)

    j = 0
    for i in range(start_tid_bench, end_tid_bench):
      tid = tids[i]
      pin_thread_to_core(tid, core_assignments[bench_id][j])
      j += 1

######### FUNCTION 1: CREATE SHARED MEMORY ###################################################################
def create_and_attach_shared_memory():
  SHM_SIZE = ctypes.sizeof(SharedData)
  shm = posix_ipc.SharedMemory(SHM_NAME, flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR, size=SHM_SIZE, mode=0o666)
  mapfile = mmap.mmap(shm.fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
  return shm, mapfile

######### FUNCTION 2: COLLECT THREAD IDS ###################################################################
def get_tids_for_benchmark(shared_memory_name, num_threads):
  try:
    # Access the shared memory segment
    shm = posix_ipc.SharedMemory(shared_memory_name, flags=posix_ipc.O_RDONLY)

    # Map the shared memory to the process address space
    shm_size = ctypes.sizeof(SharedData)  
    mapped_memory = mmap.mmap(shm.fd, shm_size, mmap.MAP_SHARED, mmap.PROT_READ)

    shm.close_fd()  # Optional: Close fd after mmap
    shared_data = SharedData.from_buffer_copy(mapped_memory)
    
    # Retrieve the thread IDs from the shared memory
    thread_ids = [shared_data.thread_ids[i] for i in range(num_threads)]
    mapped_memory.close()
    
    return thread_ids    
    
  except posix_ipc.PermissionsError as e:
    print(f"Permission error: {e}")
    return None
  except Exception as e:
    print(f"Error accessing shared memory: {e}")
    return None

######### FUNCTION 3: LAUNCH BENCHMARKS ######################################################################
def launch_benchmarks():
  processes = []
  pid_to_index = {}
  i = 0
  for command, affinity in benchmark_configs:
    full_command = f"taskset -c {affinity} numactl -m 0 {command}"
    cmd = shlex.split(full_command)

    benchmark_env = os.environ.copy()
    benchmark_env["LD_PRELOAD"] = LD_PRELOAD
    benchmark_env["OMP_NUM_THREADS"] = str(THREADS_PER_APP)  
    benchmark_env["BENCHMARK_ID"] = str(i)
      
    out = open(current_dir + "/logs/stdout" + str(i) + ".log", "w")
    err = open(current_dir + "/logs/stderr" + str(i) + ".log", "w")

    proc = subprocess.Popen(
      cmd,
      env=benchmark_env,
      stdout=out,
      stderr=err,
      bufsize=1,
      text=True
    )

    processes.append(proc)
    pid_to_index[proc.pid] = i
    i += 1

  return processes, pid_to_index

######### FUNCTION 4: MONITOR APPS ###########################################################################
LLC_HITS = [0.0] * MAX_BENCHMARKS
LLC_MISSES = [0.0] * MAX_BENCHMARKS

def monitor():
  # print("Collecting performance data thread-wise:")
  instructions = []
  cycles = []

  for bench_id in range(MAX_BENCHMARKS):
    tids = (get_tids_for_benchmark(SHM_NAME, TOTAL_TIDS))

    # Initialize the total instructions for the current benchmark
    total_instr = 0
    total_cycl = 0

    total_hits = 0
    total_miss = 0

    # Loop through the thread IDs and aggregate the instructions
    for i in range(THREADS_PER_APP):
      core_id = core_assignments[bench_id][i]
      instr = profiler.getEventOnCore(core_id, 0)  # Instructions
      cycl = profiler.getEventOnCore(core_id, 1)

      hits = profiler.getEventOnCore(core_id, 3)
      miss = profiler.getEventOnCore(core_id, 4)

      total_instr += instr
      total_cycl += cycl

      total_hits += hits
      total_miss += miss

    # Update the global instructions array with the total instructions for the benchmark
    if len(INSTRUCTIONS) <= bench_id:
      INSTRUCTIONS.append(0)
      CYCLES.append(0)

    instructions.append(total_instr)
    cycles.append(total_cycl)
    
    INSTRUCTIONS[bench_id] += total_instr
    CYCLES[bench_id] += total_cycl

    LLC_HITS[bench_id] += total_hits
    LLC_MISSES[bench_id] += total_miss

    if(total_cycl != 0):
      IPC_sample[bench_id] = total_instr / total_cycl
  
  energy = profiler.getEventOnSocket(0, 2)
  return instructions, cycles, energy

######### FUNCTION 5: MAIN FUNCTION #########################################################################

def throughput(ips_values):
  return np.prod(ips_values) ** (1 / len(ips_values))

######### FUNCTION 6: MAIN FUNCTION #########################################################################
def main():
  get_benchmark_configs()
  assign_cores()
  ENERGY = 0

  ## CREATING AND ATTACHING SHARED MEMORY
  print("Attaching shared memory...")
  shm, mapfile = create_and_attach_shared_memory()

  print("Launching benchmarks...")
  processes, pid_to_index = launch_benchmarks()
  start_time = time.time()

  time.sleep(5)

  get_tids_for_benchmark(SHM_NAME, TOTAL_TIDS)
  print("Tids collected successfully")
  print()

  pin_thread_to_cores()
  print("Threads pinned successfully")

  ## INITIALIZE LIBPFM PROFILER
  profiler.libpfm_init()

  flag = 0

  try:
    print("Monitoring launched benchmarks...")
    while processes:
      time.sleep(SLEEP_INTERVAL)

      profiler.accumulate_pmcs()
      profiler.accumulate_energy()

      instructions, cycles, energy = monitor() 
      ENERGY += energy

      for proc in processes[:]:
        if proc.poll() is not None:  # Process finished
          print(f"Benchmark PID {proc.pid} finished.")
          processes.remove(proc)
          end_time = time.time()
          bench_index = pid_to_index[proc.pid]
          TIME[bench_index] = end_time - start_time

  finally:
    print("Waiting for any remaining benchmarks to finish...")
    for proc in processes:
      proc.wait()

  ## FINALIZE LIBPFM PROFILER
  profiler.libpfm_finalize()

  print("-----\nmkdir timedrun fake")
  print()
  print("============================ Tabulate Statistics ============================")
  for bench_id in range(MAX_BENCHMARKS):
    IPC = INSTRUCTIONS[bench_id] / CYCLES[bench_id]
    IPC_FINAL[bench_id] = IPC
    SPEEDUP = throughput(IPC_FINAL)

  for bench_id in range(MAX_BENCHMARKS):
    print(f"Instructions_{bench_id}  Cycles_{bench_id}  IPC_{bench_id}  Hits_{bench_id}  Misses_{bench_id}  Time_{bench_id}  ", end="")
  print(f"Energy  Speedup")

  for bench_id in range(MAX_BENCHMARKS):
    print(f"{INSTRUCTIONS[bench_id]}  {CYCLES[bench_id]}  {IPC_FINAL[bench_id]:.5f}  {LLC_HITS[bench_id]}  {LLC_MISSES[bench_id]}  {TIME[bench_id]:.5f}  ", end="")
  print(f"{ENERGY:.5f}  {SPEEDUP:.5f}")
  
  print("=============================================================================")
  print(f"===== Total Time in {(time.time() - start_time):.5f} sec =====")

  ## SHARED MEMORY UNLINKING
  mapfile.close()
  shm.unlink()

  handle = profiler._handle  # Get the raw handle
  dlclose = ctypes.CDLL(ctypes.util.find_library('dl')).dlclose
  dlclose.argtypes = [ctypes.c_void_p]
  dlclose.restype = ctypes.c_int
  dlclose(handle)

######### CALL MAIN FUNCTION ################################################################################
if __name__ == "__main__":
  main()
