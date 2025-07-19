#!/bin/bash

BENCHMARK=$1
shift
BENCH_ARGS="$@"

mkdir -p logs/$BENCHMARK

# Find the binary
BIN_FILE=$(find ./benchmarks/$BENCHMARK/bin -type f -iname '*test' 2>/dev/null | head -n 1)

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

if [[ "$BIN_FILE" == *"/bin/"* ]]; then
  cd_target=$(dirname "$BIN_DIR")
  binary_rel_path="bin/$BIN_NAME"
else
  cd_target="$BIN_DIR"
  binary_rel_path="./$BIN_NAME"
fi

# Group B counters
perf stat -e instructions,page-faults,branch-misses,L1-dcache-load-misses -I 50 -a -x, -o logs/$BENCHMARK/group_B.csv -- bash -c "cd $cd_target && timeout 1s numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $binary_rel_path $BENCH_ARGS"
