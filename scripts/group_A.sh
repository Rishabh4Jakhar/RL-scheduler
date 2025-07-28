#!/bin/bash

BENCHMARK=$1
shift                           # Shift to access all remaining args as benchmark-specific args
BENCH_ARGS="$@"                 # All remaining arguments

mkdir -p logs/$BENCHMARK

# Try bin/ directory first
BIN_FILE=$(find ./benchmarks/$BENCHMARK/bin -type f -iname '*test' 2>/dev/null | head -n 1)

# Fallback: find in benchmark root (for CoHMM, etc.)
if [ -z "$BIN_FILE" ]; then
  BIN_FILE=$(find ./benchmarks/$BENCHMARK -maxdepth 1 -type f \( -iname '*test' -o -iname 'cohmm' -o -iname 'miniqmc' \) | head -n 1)
fi

if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found for $BENCHMARK"
  exit 1
fi

BIN_DIR=$(dirname "$BIN_FILE")
BIN_NAME=$(basename "$BIN_FILE")

echo "[*] Using binary: $BIN_FILE"

# Decide working directory and relative path to binary
if [[ "$BIN_FILE" == *"/bin/"* ]]; then
  cd_target=$(dirname "$BIN_DIR")
  binary_rel_path="bin/$BIN_NAME"
else
  cd_target="$BIN_DIR"
  binary_rel_path="./$BIN_NAME"
fi

# Final execution
perf stat -e duration_time,task-clock,context-switches,cpu-cycles -I 50 -a -x, -o logs/$BENCHMARK/group_A.csv -- bash -c "cd $cd_target && numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $binary_rel_path $BENCH_ARGS"
