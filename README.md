## PROFILER + DATA COLLECTION

This profiler setup is designed to log hardware performance counters while running various benchmarks (LULESH, AMG, miniQMC, miniMD, Quicksilver). Logs are collected via `perf stat`, parsed, and merged into structured CSV datasets for RL Model Training (Yet to be implemented).

---

### SYSTEM PREREQUISITES

```bash
numactl --hardware  # To inspect NUMA layout
chmod +x requirements.sh
./requirements.sh   # Installs perf, etc.
```

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

---

### SCRIPT OVERVIEW

All scripts are located in the `scripts/` directory.

| Script               | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| `collect_dataset.sh` | Automates running benchmarks with `perf`, logging, and merging counters. |
| `group_x.sh`         | Used to get 4 counters and store the raw logs                            |
| `merge_logs.py`      | Merges raw `perf stat` logs into a unified CSV per benchmark run.        |
| `test.sh`            | Runs the profiler on default configurations for sanity checking.         |

---

### COLLECTING DATA

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

* Run the binary with a timeout of 1 second each
* Capture 12 hardware performance counters using `perf`
* Store raw logs in `logs/<benchmark_name>/`
* Merge and final output: `logs/<benchmark_name>/<benchmark_name>_dataset.csv`

---

### OUTPUT LOGS

Logs are stored under:

```
profiler/logs/<benchmark_name>/
```

Includes:

* `*.log`: Raw output from `perf`
* `*_dataset.csv`: Final merged dataset with all counters
* Any temporary `.yaml` configs (if applicable)

---

### ADDING A NEW BENCHMARK

To include a new benchmark:

1. Place the binary in: `benchmarks/<benchmark>/bin/<benchmark>_test`
2. Ensure it runs with default config or input file
3. Add a case entry in `collect_dataset.sh` to recognize the new benchmark
4. Re-run: `./collect_dataset.sh <benchmark>`

Make sure the benchmark runs within the 1-second timeout or adjust it in the script.

---

### NOTES

* `perf stat` logs are collected in CSV format using `-x,`
* We use 12 counters split across `group_A`, `group_B`, and `group_C` scripts
* Time alignment and thread pinning is handled in `merge_logs.py`
