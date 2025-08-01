#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
BENCHMARK=$1
shift
BENCH_ARGS="$@"

mkdir -p logs/$BENCHMARK

# Primary: bin/ folder
BIN_FILE=$(find "$PROJECT_ROOT/benchmarks/$BENCHMARK/bin" -type f -iname '*test' 2>/dev/null | head -n 1)

# Secondary: benchmark root
if [ -z "$BIN_FILE" ]; then
  BIN_FILE=$(find "$PROJECT_ROOT/benchmarks/$BENCHMARK" -maxdepth 1 -type f -iname '*test' 2>/dev/null | head -n 1)
fi

# Special case: known exceptions
if [ -z "$BIN_FILE" ]; then
  case "$BENCHMARK" in
    "XSBench")
      BIN_FILE="$PROJECT_ROOT/benchmarks/XSBench/openmp-threading/XSBench_test"
      ;;
    "simplemoc")
      BIN_FILE="$PROJECT_ROOT/benchmarks/simplemoc/src/SimpleMOC_test"
      ;;
    "simplemoc_custom1")
      BIN_FILE="$PROJECT_ROOT/benchmarks/simplemoc/src/SimpleMOC_test"
      ;;
    "XSBench_custom1")
      BIN_FILE="$PROJECT_ROOT/benchmarks/XSBench/openmp-threading/XSBench_test"
      ;;
    "XSBench_custom2")
      BIN_FILE="$PROJECT_ROOT/benchmarks/XSBench/openmp-threading/XSBench_test"
      ;;
    "XSBench_custom3")
      BIN_FILE="$PROJECT_ROOT/benchmarks/XSBench/openmp-threading/XSBench_test"
      ;;
    "simplemoc_custom2")
      BIN_FILE="$PROJECT_ROOT/benchmarks/simplemoc/src/SimpleMOC_test"
      ;;
    "simplemoc_custom3")
      BIN_FILE="$PROJECT_ROOT/benchmarks/simplemoc/src/SimpleMOC_test"
      ;;  
  esac
fi


# Final check
if [ ! -f "$BIN_FILE" ]; then
  echo "[!] Benchmark binary not found for $BENCHMARK"
  echo "[!] Tried custom locations, still not found."
  exit 1
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

# Group B counter
perf stat -e instructions,page-faults,branch-misses,L1-dcache-load-misses -I 50 -a -x, -o /home/rishabh2025/profiler/logs/$BENCHMARK/group_B.csv -- bash -c "cd $cd_target && numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 $binary_rel_path $BENCH_ARGS"
