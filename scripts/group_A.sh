#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

BENCHMARK=$1
shift
BENCH_ARGS="$@"

mkdir -p /home/rishabh2025/profiler/logs/$BENCHMARK

# ➤ Detect base benchmark name (strip _custom*)
BASE_NAME=$(echo "$BENCHMARK" | sed -E 's/_custom[0-9]*//')

# === Step 1: Find binary ===

# Primary: bin/ folder
BIN_FILE=$(find "$PROJECT_ROOT/benchmarks/$BASE_NAME/bin" -type f -iname '*test' 2>/dev/null | head -n 1)

# Secondary: benchmark root dir
if [ -z "$BIN_FILE" ]; then
  BIN_FILE=$(find "$PROJECT_ROOT/benchmarks/$BASE_NAME" -maxdepth 1 -type f -iname '*test' 2>/dev/null | head -n 1)
fi

# Fallbacks (special cases)
if [ -z "$BIN_FILE" ]; then
  case "$BASE_NAME" in
    "XSBench")
      BIN_FILE="$PROJECT_ROOT/benchmarks/XSBench/openmp-threading/XSBench_test"
      ;;
    "simplemoc")
      BIN_FILE="$PROJECT_ROOT/benchmarks/simplemoc/src/SimpleMOC_test"
      ;;
  esac
fi

# Final check
if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found for $BENCHMARK"
  exit 1
fi

echo "[*] Using binary: $BIN_FILE"

# === Step 2: Resolve working dir and relative binary path ===

if [[ "$BIN_FILE" == *"/bin/"* ]]; then
  cd_target=$(dirname "$BIN_FILE")/..
  binary_rel_path="bin/$(basename "$BIN_FILE")"
else
  cd_target=$(dirname "$BIN_FILE")
  binary_rel_path="./$(basename "$BIN_FILE")"
fi

# === Step 3: Run perf (group A) ===
perf stat -e duration_time,task-clock,context-switches,cpu-cycles -I 50 -a -x, -o "/home/rishabh2025/profiler/logs/$BENCHMARK/group_A.csv" -- bash -c "cd $cd_target && numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $binary_rel_path $BENCH_ARGS"
