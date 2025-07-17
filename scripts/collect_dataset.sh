#!/bin/bash

#bash scripts/test.sh

BENCHMARK=$1

if [ -z "$BENCHMARK" ]; then
  echo "Usage: $0 <benchmark>"
  exit 1
fi
echo "[*] Running for benchmark: $BENCHMARK"

mkdir -p logs/$BENCHMARK

bash scripts/test.sh
sleep 0.1
bash scripts/group_A.sh $BENCHMARK
sleep 0.1
bash scripts/group_B.sh $BENCHMARK
sleep 0.1
bash scripts/group_C.sh $BENCHMARK
sleep 0.1
python3 scripts/merge_logs.py $BENCHMARK
#mv logs/merged.csv logs/$BENCHMARK/${BENCHMARK}_dataset.csv
echo "[✓] Dataset for $BENCHMARK saved as logs/$BENCHMARK/${BENCHMARK}_dataset.csv"