#!/bin/bash

#
# Performance Dataset Collection Script
#
# This script collects comprehensive performance data for HPC benchmarks by running
# them under different resource allocation scenarios (Group A, B, C). It measures
# both solo execution time and detailed performance counters for RL training.
#
# Usage: ./collect_dataset.sh <benchmark> [args...]
#        ./collect_dataset.sh XSBench
#        ./collect_dataset.sh custom_job (uses custom_commands.txt)
#
# Output: Creates logs/<benchmark>/<benchmark>_dataset.csv with performance data
#

BENCHMARK=$1
shift
BENCH_ARGS="$@"

if [ -z "$BENCHMARK" ]; then
  echo "Usage: $0 <benchmark>"
  exit 1
fi

echo "[*] Running for benchmark: $BENCHMARK"
mkdir -p /home/rishabh2025/profiler/logs/$BENCHMARK

# ===================================================================
# Step 0: Custom Command Detection
# ===================================================================
# Check if benchmark has custom execution parameters defined
CUSTOM_FILE="custom_commands.txt"
if [ -f "$CUSTOM_FILE" ]; then
  while IFS= read -r line; do
    job=$(echo "$line" | cut -d'::' -f1)
    cmd=$(echo "$line" | cut -d'::' -f2-)
    if [ "$job" == "$BENCHMARK" ]; then
      echo "[*] Using custom command for $BENCHMARK"
      CMD="$cmd"
      break
    fi
  done < "$CUSTOM_FILE"
fi

# ===================================================================
# Step 1: Solo Runtime Baseline Measurement
# ===================================================================
# Measure execution time without performance counter overhead
# This provides the baseline for RWI calculations
echo "[*] Measuring SOLO run time for $BENCHMARK"
START_NS=$(date +%s%N)

if [ -n "$CMD" ]; then
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK --custom "$CMD"
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK --custom "$CMD"
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK --custom "$CMD"
else
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK $BENCH_ARGS
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK $BENCH_ARGS
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK $BENCH_ARGS
fi

END_NS=$(date +%s%N)
SOLO_NS=$((END_NS - START_NS))
SOLO_TIME=$(awk "BEGIN { printf \"%.3f\", $SOLO_NS / 1000000000 }")
export SOLO_TIME
echo "[*] Solo runtime = ${SOLO_TIME}s"

# ===================================================================
# Step 2: Detailed Performance Counter Collection
# ===================================================================
# Re-run with performance monitoring enabled to collect training data
# Sleep intervals prevent interference between measurement runs
bash /home/rishabh2025/profiler/scripts/test.sh
sleep 0.1
if [ -n "$CMD" ]; then
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK --custom "$CMD"
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK --custom "$CMD"
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK --custom "$CMD"
else
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK $BENCH_ARGS
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK $BENCH_ARGS
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK $BENCH_ARGS
fi

# ===================================================================
# Step 3: Dataset Generation and Consolidation
# ===================================================================
# Merge all collected performance data into a single CSV dataset
# Includes solo_time baseline for RWI reward calculations
python3 /home/rishabh2025/profiler/scripts/merge_logs.py $BENCHMARK $SOLO_TIME
echo "Dataset for $BENCHMARK saved as logs/$BENCHMARK/${BENCHMARK}_dataset.csv"
