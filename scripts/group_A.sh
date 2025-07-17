#!/bin/bash

BENCHMARK=$1
INPUT_ID=${2:-1}
mkdir -p logs/$BENCHMARK

# Try bin/ directory first
BIN_FILE=$(find ./benchmarks/$BENCHMARK/bin -type f -iname '*test' 2>/dev/null | head -n 1)

# Fallback: find in benchmark root (for CoHMM, etc.)
if [ -z "$BIN_FILE" ]; then
  BIN_FILE=$(find ./benchmarks/$BENCHMARK -maxdepth 1 -type f \( -iname '*test' -o -iname 'cohmm' \) | head -n 1)
fi

# Final check
if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found for $BENCHMARK"
  exit 1
fi

BIN_DIR=$(dirname "$BIN_FILE")
BIN_NAME=$(basename "$BIN_FILE")

echo "[*] Using binary: $BIN_FILE"

# Determine relative binary path from BIN_DIR
if [[ "$BIN_FILE" == *"/bin/"* ]]; then
  # Binary is in bin/, so we assume benchmark can run from parent folder
  cd_target=$(dirname "$BIN_DIR")  # e.g., ./benchmarks/CoMD
  binary_rel_path="bin/$BIN_NAME"
else
  # Binary is directly in the benchmark folder
  cd_target="$BIN_DIR"             # e.g., ./benchmarks/CoHMM
  binary_rel_path="./$BIN_NAME"
fi

perf stat -e duration_time,task-clock,context-switches,cpu-cycles -I 50 -a -x, -o logs/$BENCHMARK/group_A.csv -- bash -c "cd $cd_target && timeout 1s numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $binary_rel_path $INPUT_ID"
