#!/bin/bash

BENCHMARK=$1
mkdir -p logs/$BENCHMARK

BIN_DIR="./benchmarks/$BENCHMARK/bin"
BIN_FILE=$(find ./benchmarks/$BENCHMARK -type f -iname '*test' | head -n 1)

if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found in $BIN_DIR"
  exit 1
fi
perf stat -e instructions,page-faults,branch-misses,L1-dcache-load-misses -I 50 -a -x, -o logs/$BENCHMARK/group_B.csv -- bash -c "timeout 1s numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $BIN_FILE"
