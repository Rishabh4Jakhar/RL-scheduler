#!/usr/bin/env python3
"""
Usage:
    python sim_custom.py [W] [Cmax]
    
    W: Number of jobs to schedule (default: 6)
    Cmax: Maximum concurrent jobs per slot (default: 4)
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv

# ----------------------
# 1) Parse command line arguments
# ----------------------
# Parse command line arguments for workload size and concurrency limit
W    = int(sys.argv[1]) if len(sys.argv) > 1 else 6      # Number of jobs to schedule
Cmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4      # Max concurrent jobs per scheduling slot

# Display usage instructions to the user
print(f"\nEnter {W} job names. You can also provide custom jobs like:")
print("XSBench_custom1 --cmd /full/path/to/XSBench -G nuclide -g 7000 -l 200 -p 350000 -t 12")
print("If no --cmd is given, default benchmark logs are used (e.g. 'AMG')\n")

# Initialize data structures for job collection
jobs            = []  # List of job names
custom_commands = {}  # Dictionary mapping custom job names to their commands

# Collect job specifications from user input
for i in range(W):
    line = input(f"▶ Enter job {i+1}: ").strip()
    if not line:
        print("Empty input, exiting.")
        sys.exit(1)
    
    # Parse custom command syntax: "jobname --cmd command_with_args"
    if "--cmd" in line:
        name, cmd = line.split("--cmd", 1)
        name = name.strip()
        cmd  = cmd.strip()
        jobs.append(name)
        custom_commands[name] = cmd
    else:
        # Default benchmark job (uses existing logs)
        jobs.append(line)

# ----------------------
# 2) Ensure performance logs exist for all jobs
# ----------------------
# Define paths for log collection
LOG_BASE = os.path.expanduser("~/profiler/logs")  # Base directory for all benchmark logs
COLLECT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "scripts", "collect_dataset.sh"))  # Log collection script

# Check and generate missing performance logs for each job
for job in jobs:
    ds_dir  = os.path.join(LOG_BASE, job)           # Job-specific log directory
    ds_file = os.path.join(ds_dir, f"{job}_dataset.csv")  # Expected dataset file
    
    if os.path.exists(ds_file):
        print(f"Found existing logs for {job}")
        continue

    print(f"[!] Logs not found for {job}, running collect_dataset.sh…")
    os.makedirs(ds_dir, exist_ok=True)

    # For custom jobs, append command to custom_commands.txt
    if job in custom_commands:
        # Write custom command specification to file for the collection script
        cc_path = os.path.join(os.path.dirname(__file__), "custom_commands.txt")
        with open(cc_path, "a") as f:
            f.write(f"{job}::{custom_commands[job]}\n")

    # Execute the log collection script
    ret = subprocess.call(f"bash {COLLECT} {job}", shell=True)
    if ret != 0 or not os.path.exists(ds_file):
        print(f"Failed to generate logs for {job}")
        sys.exit(1)

    print(f"Logs generated for {job}")

# ----------------------
# 3) Load performance datasets into DataFrames
# ----------------------
# Load performance data for each job into pandas DataFrames
dfs = []
for job in jobs:
    df = pd.read_csv(os.path.join(LOG_BASE, job, f"{job}_dataset.csv"))
    dfs.append(df)

# ----------------------
# 4) Initialize RL model and scheduling environment
# ----------------------
# Load the pre-trained PPO model for job scheduling decisions
model = PPO.load(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "ppo_rwi_scheduler.zip")))

# Create the multi-job scheduling environment with loaded performance data
env   = MultiJobSchedulingEnv(dfs, num_jobs=W, Cmax=Cmax,
                              use_all_actions=False,    # Use simplified action space
                              shuffle_on_reset=False)   # Maintain job order consistency

# Initialize the environment and get initial observation
obs, _ = env.reset()

# ----------------------
# 5) Run RL-based scheduling simulation
# ----------------------
# Initialize scheduling matrices and tracking variables
S         = np.zeros((W, W), dtype=int)  # Co-scheduling matrix: S[i,j] = 1 if job j scheduled in slot i
R         = np.zeros(W, dtype=int)       # Resource allocation vector: 1=Core, 2=Thread
scheduled = [False] * W                  # Track which jobs have been scheduled
row       = 0                           # Current scheduling slot

# Main scheduling loop: iterate through time slots until all jobs are scheduled
while row < W and not all(scheduled):
    # Get RL model's action prediction based on current environment state
    action, _ = model.predict(obs, deterministic=True)
    
    # Parse action into scheduling matrix and resource allocation components
    s_flat    = action[: W*W]           # Flattened scheduling decisions
    r_vec     = action[W*W : W*W + W]   # Resource allocation preferences

    # Convert flattened scheduling decisions to selection vector
    sel = s_flat.reshape(W, W).sum(axis=0)  # Sum across rows to get job selection
    sel = (sel > 0).astype(int)             # Binary selection: 1 if selected, 0 otherwise
    
    # Apply scheduling constraints
    # Constraint 1: Don't reschedule already completed jobs
    for j in range(W):
        if scheduled[j]:
            sel[j] = 0
    
    # Constraint 2: Enforce maximum concurrency limit (Cmax)
    if sel.sum() > Cmax:
        # Select top Cmax jobs based on resource preference scores
        order = np.argsort(-r_vec)      # Sort jobs by preference (descending)
        keep  = order[:Cmax]            # Keep only top Cmax jobs
        mask  = np.zeros(W, dtype=int)
        mask[keep] = 1
        sel = sel * mask                # Apply selection mask

    # Record scheduling decisions for current slot
    S[row, :] = sel
    for j in range(W):
        if sel[j]:
            R[j] = int(r_vec[j]) + 1  # Map to resource type: 1=Core, 2=Thread

    # Prepare action for environment step (convert to expected format)
    step_act = np.zeros(W*W + W, dtype=int)
    for j in range(W):
        if sel[j]:
            step_act[j*(W+1)]     = 1         # Set scheduling decision
            step_act[W*W + j]     = r_vec[j]  # Set resource allocation

    # Execute scheduling step in environment
    obs, _, term, trunc, _ = env.step(step_act)

    # Update job completion status
    for j in range(W):
        if sel[j]:
            scheduled[j] = True
    
    # Move to next scheduling slot
    row += 1
    
    # Check for early termination conditions
    if term or trunc:
        break

# ----------------------
# 6) Display scheduling results
# ----------------------
# Display the final scheduling matrices
print("\n🧩 Final Co-Scheduling Matrix S:")
print("   (Rows = time slots, Columns = jobs, 1 = scheduled)")
print(S)

print("\n🧩 Final Resource Vector R:")
print("   (1 = Core mode, 2 = Thread mode)")
print(R)

# Display human-readable scheduling groups
print("\n📋 Co-Scheduling Groups:")
for i in range(W):
    group = np.where(S[i] == 1)[0]  # Find jobs scheduled in slot i
    if len(group) == 0:
        continue
    
    # Map resource codes to human-readable names
    modes = ["Core" if R[j] == 1 else "Thread" for j in group]
    names = [jobs[j] for j in group]
    print(f"  ▪️ Slot {i}: {names} → {modes}")

print("\nScheduling simulation completed successfully.")
