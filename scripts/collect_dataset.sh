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

# === Step 0: Check if a custom command exists ===
CUSTOM_FILE="custom_commands.txt"
if [ -f "$CUSTOM_FILE" ]; then
  while IFS= read -r line; do
    job=$(echo "$line" | cut -d'::' -f1)
    cmd=$(echo "$line" | cut -d'::' -f2-)
    if [ "$job" == "$BENCHMARK" ]; then
      echo "[*] Using custom command for $BENCHMARK"
      CMD="$cmd"
      break
    fi
  done < "$CUSTOM_FILE"
fi

# === Step 1: Measure solo runtime ===
echo "[*] Measuring SOLO run time for $BENCHMARK"
START_NS=$(date +%s%N)

if [ -n "$CMD" ]; then
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK --custom "$CMD"
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK --custom "$CMD"
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK --custom "$CMD"
else
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK $BENCH_ARGS
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK $BENCH_ARGS
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK $BENCH_ARGS
fi

END_NS=$(date +%s%N)
SOLO_NS=$((END_NS - START_NS))
SOLO_TIME=$(awk "BEGIN { printf \"%.3f\", $SOLO_NS / 1000000000 }")
export SOLO_TIME
echo "[*] Solo runtime = ${SOLO_TIME}s"

# === Step 2: Repeat with sleep for dataset logging ===
bash /home/rishabh2025/profiler/scripts/test.sh
sleep 0.1
if [ -n "$CMD" ]; then
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK --custom "$CMD"
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK --custom "$CMD"
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK --custom "$CMD"
else
  bash /home/rishabh2025/profiler/scripts/group_A.sh $BENCHMARK $BENCH_ARGS
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_B.sh $BENCHMARK $BENCH_ARGS
  sleep 0.1
  bash /home/rishabh2025/profiler/scripts/group_C.sh $BENCHMARK $BENCH_ARGS
fi

# === Step 3: Merge logs ===
python3 /home/rishabh2025/profiler/scripts/merge_logs.py $BENCHMARK $SOLO_TIME
echo "Dataset for $BENCHMARK saved as logs/$BENCHMARK/${BENCHMARK}_dataset.csv"
