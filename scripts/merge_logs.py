import pandas as pd
import sys

def load_perf(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue  # malformed row
            try:
                timestamp = float(parts[0])
                value = float(parts[1])
                event = parts[3].strip()
                rows.append((int(timestamp * 1000), event, value))  # convert to ms
            except ValueError:
                continue  # skip non-numeric rows
    df = pd.DataFrame(rows, columns=["time", "event", "value"])
    return df.pivot(index="time", columns="event", values="value")

# Load logs
df_A = load_perf(sys.argv[1])
df_B = load_perf(sys.argv[2])
df_C = load_perf(sys.argv[3])

# Merge all groups on time
merged = pd.concat([df_A, df_B, df_C], axis=1)
merged = merged.reset_index()

# Add metadata (customize as needed)
merged["benchmark"] = "CoMD"
merged["socket"] = 0
merged["core_list"] = "0,2,4,6,8,10,12,14,16,18"

# Save final dataset
merged.to_csv("logs/final_dataset.csv", index=False)
