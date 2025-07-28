#!/bin/bash

# List of all 10 benchmark names (matching folder names under benchmarks/)
BENCHMARKS=(
  AMG
  XSBench
  minTally
  simplemoc
  hpccg
  CoHMM
  CoMD
  miniMD
  miniQMC
  Quicksilver
)

echo "==============================="
echo " Starting batch profiling..."
echo "==============================="

for BENCH in "${BENCHMARKS[@]}"; do
  echo -e "\n🔧 Profiling benchmark: $BENCH"
  ./collect_dataset.sh $BENCH
done

echo -e "\n✅ All 10 benchmarks profiled and datasets saved under logs/<BENCH>/"
