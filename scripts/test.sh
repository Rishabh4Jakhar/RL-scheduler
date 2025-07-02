#!/bin/bash

declare -A job_queue_dict=(
  [HP]="../benchmarks/hpccg/HPCCG_test"
  [CMD]="../benchmarks/CoMD/bin/CoMD_test -x 16 -y 16 -z 16 -N 10000 -D 10"
  [MT]="../benchmarks/minTally/bin/mintally_test"
  [XS]="../benchmarks/XSBench/openmp-threading/XSBench_test -G nuclide -g 5000 -l 100 -p 450000 -t 20"
  [SM]="../benchmarks/simplemoc/src/SimpleMOC_test -i /home/$USER/profiler/benchmarks/simplemoc/src/custom_input_2.in -t 20"
)

PYTHON_SCRIPT="../src/main.py"

# Declare an array of combinations (pairs of keys to pass)
combinations=("HP")                                       # Add benchmarks here
ranges=("0,2,4,6,8,10,12,14,16,18,20")                    # Add thread affinities here; only physical cores

## For multiple benchmark testing at same time:
# combinations=("Bench1" "Bench2")                                       
# ranges=("Affinity1 separated by commas" "Affinity2 separated by commas")

# Prepare the corresponding ranges for the combination
range_str=""
for j in {0..0}; do
  range_str+="${ranges[$j]} "  # Collect the ranges for this combination
done

# Loop through each combination and prepare the environment variables
for i in {0..0}
do
  for i in "${!combinations[@]}"; do
    combo="${combinations[$i]}"
    
    # Prepare the environment variables for BENCHMARK_NAMES and BENCHMARK_BINARIES
    benchmark_names_str=""
    benchmark_binaries_str=""

    # Loop over the keys in the combination and append the corresponding values to the environment variables
    for key in $combo; do
      if [[ -n "${job_queue_dict[$key]}" ]]; then
        benchmark_names_str="$benchmark_names_str,$key"
        benchmark_binaries_str="$benchmark_binaries_str,${job_queue_dict[$key]}"
      else
        echo "Key $key not found in job_queue_dict"
        exit 1
      fi
    done
    # remove leading commas
    benchmark_names_str="${benchmark_names_str#,}"
    benchmark_binaries_str="${benchmark_binaries_str#,}"

    log_prefix=$(echo "$combo" | tr ' ' '_')

    log_suffix="DEFAULT"
    RANGES="$range_str" BENCHMARK_NAMES="$benchmark_names_str" BENCHMARK_BINARIES="$benchmark_binaries_str" python3 "$PYTHON_SCRIPT" >> ../logs/${log_prefix}.3096.1024.dino.policy-$log_suffix.log

    done

done

rm *.yaml