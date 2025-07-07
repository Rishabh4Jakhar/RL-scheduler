#!/bin/bash
bash scripts/group_A.sh
bash scripts/group_B.sh
bash scripts/group_C.sh

python3 scripts/merge_logs.py logs/group_A.csv logs/group_B.csv logs/group_C.csv
