import pandas as pd

# Replace with your actual paths
group_A = pd.read_csv("./logs/group_A.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])
group_B = pd.read_csv("./logs/group_B.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])
group_C = pd.read_csv("./logs/group_C.csv", comment='#', header=None,
                      names=['time', 'value', 'unit', 'event', 'raw', 'percent', 'rate', 'extra'])


# Number of counters per timestamp (perf group)
N = 4

# Split each into chunks
def extract_chunks(df):
    chunks = []
    for i in range(0, len(df), N):
        chunk = df.iloc[i:i+N].reset_index(drop=True)
        chunks.append(chunk)
    return chunks

chunks_A = extract_chunks(group_A)
chunks_B = extract_chunks(group_B)
chunks_C = extract_chunks(group_C)

# Sanity check
min_len = min(len(chunks_A), len(chunks_B), len(chunks_C))
print(f"Merging {min_len} batches...")

# Merge them by position
merged_rows = []

for i in range(min_len):
    row = {
        'time': float(chunks_A[i].loc[0, 'time']),
        'benchmark': "CoMD",
        'socket': 0,
        'core_list': "0,2,4,6,8,10,12,14,16,18"
    }
    # Add counters from each group
    for group, label in zip([chunks_A, chunks_B, chunks_C], ['A', 'B', 'C']):
        for _, row_data in group[i].iterrows():
            row[row_data['event']] = float(row_data['value']) if pd.notna(row_data['value']) else None
    merged_rows.append(row)

# Create final DataFrame
final_df = pd.DataFrame(merged_rows)

# Reorder columns (optional)
desired_order = ['time', 'duration_time', 'task-clock', 'context-switches', 'cpu-cycles',
                 'instructions', 'page-faults', 'branch-misses', 'L1-dcache-load-misses',
                 'LLC-load-misses', 'L1-icache-load-misses', 'dTLB-load-misses', 'iTLB-load-misses',
                 'benchmark', 'socket', 'core_list']

final_df = final_df[desired_order]

# Save to CSV
final_df.to_csv("./logs/merged_output.csv", index=False)
print("✅ Merged file saved as merged_output.csv")
