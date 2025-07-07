#!/bin/bash
perf stat -e instructions,page-faults,branch-misses,L1-dcache-load-misses -I 50 -a -x, -o logs/group_B.csv -- bash -c "timeout 3s numactl --physcpubind=0,2,4,6,8,10,12,14,16,18 ./benchmarks/CoMD/bin/CoMD_test"
