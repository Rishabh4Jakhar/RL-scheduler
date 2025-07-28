#!/bin/bash

#bash scripts/test.sh

BENCHMARK=$1
shift
BENCH_ARGS="$@"
if [ -z "$BENCHMARK" ]; then
  echo "Usage: $0 <benchmark>"
  exit 1
fi
echo "[*] Running for benchmark: $BENCHMARK"

mkdir -p logs/$BENCHMARK

# 2a) Measure solo run time
echo "[*] Measuring SOLO run time for $BENCHMARK"
START_NS=$(date +%s%N)
bash scripts/group_A.sh $BENCHMARK $BENCH_ARGS    # or call your full pipeline here
END_NS=$(date +%s%N)
SOLO_NS=$((END_NS - START_NS))
# convert to seconds with millisecond precision
SOLO_TIME=$(awk "BEGIN { printf \"%.3f\", $SOLO_NS/1000000000 }")
echo "[*] Solo runtime = ${SOLO_TIME}s"

# export for merge step
export SOLO_TIME

bash scripts/test.sh
sleep 0.1
bash scripts/group_A.sh $BENCHMARK $BENCH_ARGS
sleep 0.1
bash scripts/group_B.sh $BENCHMARK $BENCH_ARGS
sleep 0.1
bash scripts/group_C.sh $BENCHMARK $BENCH_ARGS
sleep 0.1
python3 scripts/merge_logs.py $BENCHMARK $SOLO_TIME
#mv logs/merged.csv logs/$BENCHMARK/${BENCHMARK}_dataset.csv
echo "Dataset for $BENCHMARK saved as logs/$BENCHMARK/${BENCHMARK}_dataset.csv"