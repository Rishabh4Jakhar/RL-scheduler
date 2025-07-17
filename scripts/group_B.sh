#!/bin/bash

BENCHMARK=$1
INPUT_ID=${2:-1}
mkdir -p logs/$BENCHMARK

BIN_FILE=$(find ./benchmarks/$BENCHMARK/bin -type f -iname '*test' 2>/dev/null | head -n 1)
if [ -z "$BIN_FILE" ]; then
  BIN_FILE=$(find ./benchmarks/$BENCHMARK -maxdepth 1 -type f -iname '*test' -o -iname 'cohmm' | head -n 1)
fi

if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found for $BENCHMARK"
  exit 1
fi
BIN_DIR=$(dirname "$BIN_FILE")
BIN_NAME=$(basename "$BIN_FILE")
BENCH_ROOT=$(dirname "$BIN_DIR")
#echo "[*] Using binary: $BIN_FILE"
perf stat -e instructions,page-faults,branch-misses,L1-dcache-load-misses -I 50 -a -x, -o logs/$BENCHMARK/group_B.csv -- bash -c "cd $BENCH_ROOT && timeout 1s numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 ./bin/$BIN_NAME $INPUT_ID"
