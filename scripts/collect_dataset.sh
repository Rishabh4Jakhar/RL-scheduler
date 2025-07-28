#!/bin/bash

BENCHMARK=$1
shift
BENCH_ARGS="$@"
if [ -z "$BENCHMARK" ]; then
  echo "Usage: $0 <benchmark>"
  exit 1
fi
echo "[*] Running for benchmark: $BENCHMARK"

mkdir -p logs/$BENCHMARK

# === Step 1: Measure solo runtime (excluding sleep) ===
echo "[*] Measuring SOLO run time for $BENCHMARK"
START_NS=$(date +%s%N)

# Actual benchmarking steps 
bash scripts/group_A.sh $BENCHMARK $BENCH_ARGS
bash scripts/group_B.sh $BENCHMARK $BENCH_ARGS
bash scripts/group_C.sh $BENCHMARK $BENCH_ARGS

END_NS=$(date +%s%N)
SOLO_NS=$((END_NS - START_NS))
SOLO_TIME=$(awk "BEGIN { printf \"%.3f\", $SOLO_NS / 1000000000 }")
echo "[*] Solo runtime = ${SOLO_TIME}s"

# Export to pass into Python
export SOLO_TIME

# === Step 2: Repeat with sleep delays for normal dataset ===
bash scripts/test.sh
sleep 0.1
bash scripts/group_A.sh $BENCHMARK $BENCH_ARGS
sleep 0.1
bash scripts/group_B.sh $BENCHMARK $BENCH_ARGS
sleep 0.1
bash scripts/group_C.sh $BENCHMARK $BENCH_ARGS
sleep 0.1

# === Step 3: Merge logs, inject SOLO_TIME into dataset ===
python3 scripts/merge_logs.py $BENCHMARK $SOLO_TIME
echo "Dataset for $BENCHMARK saved as logs/$BENCHMARK/${BENCHMARK}_dataset.csv"
