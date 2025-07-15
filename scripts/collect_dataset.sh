#!/bin/bash
bash scripts/group_A.sh
sleep 0.1
bash scripts/group_B.sh
sleep 0.1
bash scripts/group_C.sh
sleep 0.1

python3 scripts/merge_logs.py logs/group_A.csv logs/group_B.csv logs/group_C.csv
