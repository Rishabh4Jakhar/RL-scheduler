#!/usr/bin/env python3
"""
Performance Log Merger for RL Dataset Generation

This script consolidates performance counter data from multiple resource allocation
groups (A, B, C) into a unified dataset suitable for reinforcement learning training.
It processes raw perf output and creates structured CSV files with normalized
performance metrics and baseline solo execution times.

Key Functions:
- Merges performance data from 3 different resource allocation strategies
- Adds solo execution baseline for RWI calculations
- Normalizes timestamps and counter values across measurement groups
- Outputs structured CSV datasets for RL environment consumption

Input: Raw CSV files from group_A.csv, group_B.csv, group_C.csv
Output: <benchmark>_dataset.csv with consolidated performance metrics
"""

import pandas as pd
import sys 

if len(sys.argv) != 3:
    print("Usage: merge_logs.py <benchmark_name> <solo_time>")
    sys.exit(1)

benchmark = sys.argv[1]
solo_time = float(sys.argv[2])
log_path = f"/home/rishabh2025/profiler/logs/{benchmark}"

# ===================================================================
# Load Raw Performance Data 
# ===================================================================

group_A = pd.read_csv(f"{log_path}/group_A.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])
group_B = pd.read_csv(f"{log_path}/group_B.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])
group_C = pd.read_csv(f"{log_path}/group_C.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])


# Number of counters per timestamp (perf group)
N = 4

# Split performance data into temporal chunks for synchronization
def extract_chunks(df):
    """
    Split performance counter data into temporal batches
    Each chunk contains N simultaneous counter measurements
    """
    chunks = []
    for i in range(0, len(df), N):
        chunk = df.iloc[i:i+N].reset_index(drop=True)
        chunks.append(chunk)
    return chunks

chunks_A = extract_chunks(group_A)
chunks_B = extract_chunks(group_B)
chunks_C = extract_chunks(group_C)

# ===================================================================
# Data Synchronization and Merging
# ===================================================================
# Ensure all groups have same number of measurement batches
min_len = min(len(chunks_A), len(chunks_B), len(chunks_C))
print(f"Merging {min_len} batches...")

# Merge performance data by temporal position across all groups
merged_rows = []

for i in range(min_len):
    # Initialize row with metadata and baseline metrics
    row = {
        # Solo time
        'solo_time': solo_time,
        'time': float(chunks_A[i].loc[0, 'time']),
        'benchmark': f"{benchmark}",
        'socket': 0,
        'core_list': "0,2,4,6,8,10,12,14,16,18"
    }
    
    # Consolidate performance counters from all resource allocation groups
    # Each group contributes different counter types (CPU, cache, memory, etc.)
    for group, label in zip([chunks_A, chunks_B, chunks_C], ['A', 'B', 'C']):
        for _, row_data in group[i].iterrows():
            row[row_data['event']] = float(row_data['value']) if pd.notna(row_data['value']) else None
    merged_rows.append(row)

# ===================================================================
# Dataset Structuring and Output Generation
# ===================================================================
# Create final DataFrame with all consolidated performance metrics
final_df = pd.DataFrame(merged_rows)

# Standardize column ordering for consistent RL environment input
# Order: baseline metrics, execution metrics, CPU counters, cache/memory counters
desired_order = ['solo_time', 'time', 'duration_time', 'task-clock', 'context-switches', 'cpu-cycles',
                 'instructions', 'page-faults', 'branch-misses', 'L1-dcache-load-misses',
                 'LLC-load-misses', 'L1-icache-load-misses', 'dTLB-load-misses', 'iTLB-load-misses',
                 'benchmark', 'socket', 'core_list']

final_df = final_df[desired_order]

# Save consolidated dataset for RL training
final_df.to_csv(f"{log_path}/{benchmark}_dataset.csv", index=False)
