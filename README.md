## PROFILER + REINFORCEMENT LEARNING SCHEDULER

This system combines hardware performance profiling with reinforcement learning to optimize HPC job co-scheduling. It collects performance counters from various benchmarks (AMG, XSBench, miniQMC, miniMD, Quicksilver, etc.) and trains RL agent to make intelligent scheduling decisions that maximize system throughput.

<br>

The RL model achieves **approximately ~3x** throughput compared to traditional scheduling methods on custom HPC workload combinations.

![Throughput Comparison](https://github.com/Rishabh4Jakhar/RL-profiler/blob/main/rl/throughput_comparison_12.png)
<div align="center"><strong>Throughput Comparison</strong></div>
<br><br>

*Performance comparison showing significant throughput improvements of the RL scheduler over baseline methods*

### Key Capabilities

The RL scheduler learns to:
- **Co-schedule compatible jobs** for optimal resource utilization
- **Allocate resources** (Core vs Thread mode) based on workload characteristics  
- **Maximize Reward (Throughput)** compared to traditional time-sharing
- **Handle diverse HPC workloads** through curriculum learning

---

### SYSTEM PREREQUISITES

```bash
numactl --hardware  # To inspect NUMA layout
chmod +x requirements.sh
./requirements.sh   # Installs perf, etc.
```

---

### REINFORCEMENT LEARNING ARCHITECTURE

#### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Multi-Job Environment** | `rl/multi_env.py` | Advanced environment for co-scheduling multiple jobs simultaneously |
| **Training Script** | `rl/train_multi.py` | PPO training with curriculum learning across job mixes |
| **Multi-Mix Evaluation** | `rl/sim_multi.py` | Evaluate trained model on predefined job combinations |
| **Custom Simulation** | `rl/sim_custom.py` | Interactive evaluation with user-defined job mixes |
| **Result Visualization** | `rl/commpare_results.py` | Visualizes results of RL model against time-sharing and RL-FIFO

#### RL Environment Design

**Multi-Job Environment (`multi_env.py`)**
- **Observation Space**: (num_jobs × 12) concatenated performance counters for all jobs
- **Action Space**: MultiDiscrete - Two-part structure:
  - Scheduling Matrix: `num_jobs²` binary decisions for job co-scheduling
  - Resource Vector: `num_jobs` decisions (0=NoRun, 1=Core, 2=Thread)
- **Co-scheduling Logic**: Uses graph theory (connected components) to identify job groups
- **Reward**: Group-based RWI calculation with performance baselines
- **Concurrency Limit**: `Cmax=4` maximum jobs per scheduling group
- **Purpose**: Learn optimal job groupings and resource allocation for multi-job scenarios

#### Algorithm Choice: PPO (Proximal Policy Optimization)
For the RL model, PPO was selected, which is a specific RL algorithm.
**Why PPO was selected:**
- **Stability**: More stable than vanilla policy gradient methods
- **Sample Efficiency**: Better than A2C/A3C for this domain
- **Continuous Learning**: Handles curriculum learning across job mixes effectively
- **Multi-discrete Actions**: Supports complex action spaces (scheduling + resource allocation)
- **Proven Performance**: Well-established for resource allocation problems

**PPO Configuration:**
- **Policy Network**: MLP (Multi-Layer Perceptron) for both value function and policy
- **Training**: 75,000 total timesteps (15 mixes × 5,000 timesteps each)
- **Curriculum Learning**: Progressive exposure to diverse job combinations

#### Training Strategy

**Curriculum Learning Approach (`train_multi.py`):**
- **15 Random Job Mixes**: Each containing 6 different HPC benchmarks
- **Progressive Difficulty**: Exposes agent to increasing workload diversity
- **Benchmark Pool**: 10 different HPC applications with varying characteristics:
  - **AMG**
  - **XSBench**
  - **miniMD**
  - **miniQMC**
  - **Quicksilver**
  - **CoMD**
  - **CoHMM**
  - **simplemoc**
  - **hpccg**
  - **minTally**


---

### BUILDING

#### 1. Build Benchmarks

```bash
cd benchmarks
make clean && make
```

> If make fails for a specific benchmark, check that the correct build steps are followed by reading through its specific README. Logs will show details.

#### 2. Build Profiler Tools

```bash
cd profiler/tools
make clean && make
```

#### 3. Install RL Dependencies

```bash
pip install stable-baselines3 gymnasium pandas numpy scipy
```

---

### REINFORCEMENT LEARNING USAGE

#### Training a New Model

```bash
cd rl
python train_multi.py
```

This will:
- Train a PPO agent on 15 random job combinations (can be changed).
- Use curriculum learning for robust policy development
- Save the trained model as `ppo_rwi_scheduler.zip`
- Total training: ~75,000 timesteps across diverse workloads

#### Evaluating Trained Models

**Predefined Job Mix Evaluation:**
```bash
cd rl
python sim_multi.py
```
- Evaluates 6 predefined job combinations
- Calculates RWI improvements
- Shows co-scheduling matrices and resource allocations
- Displays performance gains over time-sharing

**Custom Job Mix Evaluation:**
```bash
cd rl
python sim_custom.py [num_jobs] [max_concurrent]
# Example: python sim_custom.py 6 4
```
- Interactive job specification (including custom commands)
- Supports both default benchmarks and custom executables
- Automatic log collection for missing performance data
- Real-time scheduling decisions and performance analysis

#### Key Metrics and Outputs

**Scheduling Matrices:**
- **S Matrix**: Co-scheduling decisions (jobs × time slots)
- **R Vector**: Resource allocation (1=Core, 2=Thread mode)

**Performance Counters Used:**
  `duration_time,task-clock,context-switches,cpu-cycles,instructions,page-faults,branch-misses,L1-dcache-load-misses,LLC-load-misses,L1-icache-load-misses,dTLB-load-misses,iTLB-load-misses`

---

### SCRIPT OVERVIEW

#### Data Collection Scripts (`scripts/`)

| Script               | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| `collect_dataset.sh` | Automates running benchmarks with `perf`, logging, and merging counters. |
| `group_x.sh`         | Used to get 4 counters and store the raw logs                            |
| `merge_logs.py`      | Merges raw `perf stat` logs into a unified CSV per benchmark run.        |
| `test.sh`            | Runs the profiler on default configurations for sanity checking.         |

---

### DATA COLLECTION AND RL PIPELINE

To generate datasets:

```bash
cd scripts
./collect_dataset.sh <benchmark_name>
```

Example:

```bash
./collect_dataset.sh miniMD
```

This will:
* Run the binary with different resource allocation groups (A, B, C)
* Capture 12 hardware performance counters using `perf stat`
* Measure solo execution baseline for RWI calculations
* Store raw logs in `logs/<benchmark_name>/`
* Generate final dataset: `logs/<benchmark_name>/<benchmark_name>_dataset.csv`

#### RL Training Pipeline

**Step 1: Collect Performance Data for All Benchmarks**
```bash
cd scripts
for benchmark in AMG XSBench miniMD miniQMC Quicksilver CoMD CoHMM simplemoc hpccg minTally; do
    ./collect_dataset.sh $benchmark
done
```

**Step 2: Train RL Model**
```bash
cd rl
python train_multi.py
```

**Step 3: Evaluate Trained Model**
```bash
# Evaluate on predefined mixes
python sim_multi.py

# Interactive custom evaluation
python sim_custom.py 6 4
```

#### Custom Job Integration

**Adding Custom Jobs with Specific Parameters:**

1. **Using sim_custom.py interactively:**
```bash
python sim_custom.py 6 4
# When prompted, enter:
# XSBench_custom1 --cmd /path/to/XSBench -G nuclide -g 7000 -l 200 -p 350000 -t 12
```

### OUTPUT STRUCTURE

#### Performance Logs
```
profiler/logs/<benchmark_name>/
├── group_A.csv          # Raw perf output for resource group A
├── group_B.csv          # Raw perf output for resource group B  
├── group_C.csv          # Raw perf output for resource group C
└── <benchmark>_dataset.csv  # Merged dataset for RL training
```

#### RL Model Outputs
```
profiler/rl/
├── ppo_rwi_scheduler.zip    # Trained PPO model
├── custom_commands.txt      # Custom job command definitions
└── *.png                    # Performance comparison plots (if generated)
```

#### Dataset Structure

**CSV Columns in `<benchmark>_dataset.csv`:**
- `solo_time`: Baseline execution time for RWI calculations
- `time`: Timestamp of measurement
- `duration_time`: Actual execution duration
- `task-clock`: CPU time utilization
- `context-switches`: System overhead metric
- `cpu-cycles`: Total CPU cycles consumed
- `instructions`: Total instructions executed
- `page-faults`: Memory management overhead
- `branch-misses`: Pipeline prediction failures
- `L1-dcache-load-misses`: L1 data cache misses
- `LLC-load-misses`: Last Level Cache misses
- `L1-icache-load-misses`: L1 instruction cache misses
- `dTLB-load-misses`: Data translation lookaside buffer misses
- `iTLB-load-misses`: Instruction translation lookaside buffer misses
- `benchmark`: Benchmark identifier
- `socket`: NUMA socket assignment
- `core_list`: CPU core allocation pattern


---

### ADDING NEW BENCHMARKS

#### For Data Collection

1. Place the binary in: `benchmarks/<benchmark>/bin/<benchmark>_test`
2. Ensure it runs with default config or input file
3. Add case entry in `collect_dataset.sh` to recognize the new benchmark
4. Generate performance data: `./collect_dataset.sh <benchmark>`

#### For RL Training

1. **Collect performance data** (as above)
2. **Add to benchmark pool** in `train_multi.py`:
```python
BENCHMARKS = [
    "AMG", "XSBench", "minTally", "simplemoc", "hpccg",
    "CoHMM", "CoMD", "miniMD", "miniQMC", "Quicksilver",
    "YourNewBenchmark"  # Add here
]
```
3. **Retrain model** with expanded benchmark set:
```bash
cd rl
python train_multi.py
```

#### For Custom Evaluation

**Direct specification in sim_custom.py**
```bash
python sim_custom.py 6 4
# Enter: NewBenchmark --cmd /path/to/binary --arg1 value1 --arg2 value2
```
---

### TECHNICAL NOTES AND IMPLEMENTATION DETAILS

* `perf stat` logs collected in CSV format using `-x,` delimiter
* 12 counters split strategically across `group_A`, `group_B`, and `group_C` scripts
* **Why 3 Groups**: Hardware PMU limitations require counter multiplexing
* Time alignment and thread pinning handled in `merge_logs.py`
* Solo execution baseline measured separately to avoid measurement overhead

**Reward Function Engineering:**
- **RWI Formula**: `core_alloc_ratio * (scale_factor² + duration_ratio²) + cache_miss_ratio²`
- Balances execution time, IPC, and cache efficiency

**Scalability Features:**
- **Variable Job Count**: Environment supports different num_jobs configurations
- **Benchmark Pool**: Easy addition/removal of workload types
- **Custom Commands**: Flexible parameter specification for production workloads
- **Batch Evaluation**: Efficient processing of multiple job mixes
---
