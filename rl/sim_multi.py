#!/usr/bin/env python3


import pandas as pd
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# Predefined job mix combinations for comprehensive evaluation
# Each mix contains 6 different HPC benchmarks with varying characteristics
job_mixes = [
    ["AMG", "XSBench", "minTally", "simplemoc", "hpccg", "CoHMM"],        # Mix 1: Diverse workload types
    ["miniMD", "miniQMC", "Quicksilver", "CoMD", "XSBench", "AMG"],      # Mix 2: MD/MC simulation focus
    ["simplemoc", "CoMD", "hpccg", "miniQMC", "XSBench", "minTally"],    # Mix 3: Mixed compute patterns
    ["AMG", "CoHMM", "miniMD", "Quicksilver", "CoMD", "hpccg"],          # Mix 4: Memory-intensive mix
    ["XSBench", "simplemoc", "CoMD", "Quicksilver", "CoHMM", "miniQMC"], # Mix 5: Compute-intensive mix
    ["minTally", "hpccg", "miniMD", "XSBench", "AMG", "Quicksilver"]     # Mix 6: Balanced workload
]

# Load the pre-trained PPO model for job scheduling decisions
model = PPO.load("ppo_rwi_scheduler")

# Evaluate each job mix combination
for mix_index, mix in enumerate(job_mixes):
    print(f"\n========== Evaluating Mix {mix_index + 1} ==========")
    print(f"Jobs: {mix}")

    # ----------------------
    # 1) Load performance datasets and initialize environment
    # ----------------------
    
    # Load pre-collected performance data for each job in the current mix
    dfs = [pd.read_csv(f"/home/rishabh2025/profiler/logs/{job}/{job}_dataset.csv") for job in mix]
    
    # Create multi-job scheduling environment with the performance data
    env = MultiJobSchedulingEnv(dfs, num_jobs=6, Cmax=4, 
                                use_all_actions=False,    # Use simplified action space
                                shuffle_on_reset=False)   # Maintain job order consistency
    
    # Initialize environment and get initial observation
    obs, _ = env.reset()

    # Extract environment parameters
    W = env.num_jobs     # Number of jobs to schedule (6)
    Cmax = env.Cmax      # Maximum concurrent jobs per slot (4)

    # ----------------------
    # 2) Initialize scheduling tracking variables
    # ----------------------
    
    # Track which jobs have been completed
    scheduled = [False] * W

    # Scheduling matrices:
    # S: W×W co-scheduling matrix (slot × job), 1 if job j scheduled in slot i
    # R: W-length resource allocation vector (1=Core, 2=Thread mode)
    S = np.zeros((W, W), dtype=int)
    R = np.zeros(W, dtype=int)

    # Current scheduling slot (time step)
    row = 0
    
    # ----------------------
    # 3) Execute RL-based scheduling simulation
    # ----------------------
    
    while row < W and not all(scheduled):
        # Get RL model's scheduling decision for current state
        action, _ = model.predict(obs)

        # Parse the action vector into scheduling and resource components
        s_flat = action[: W * W]           # Flattened scheduling matrix decisions
        r_vec = action[W * W : W * W + W]  # Resource allocation preferences

        # Convert flattened scheduling decisions to job selection vector
        # sel[j] = 1 if job j should be scheduled in current slot
        sel = (np.array(s_flat).reshape(W, W).sum(axis=0) > 0).astype(int)

        # Apply scheduling constraints
        # Constraint 1: Don't reschedule already completed jobs
        for j in range(W):
            if scheduled[j]:
                sel[j] = 0

        # Constraint 2: Enforce maximum concurrency limit (Cmax)
        if sel.sum() > Cmax:
            # Select top Cmax jobs based on resource preference scores
            order = np.argsort(-np.array(r_vec))  # Sort by preference (descending)
            keep = set(order[:Cmax])              # Keep only top Cmax jobs
            sel = np.array([1 if (sel[j] and j in keep) else 0 for j in range(W)])

        # Record scheduling decision for current slot
        S[row, :] = sel

        # Set resource allocation for newly scheduled jobs
        # R[j]: 1=Core mode, 2=Thread mode (default to Core if invalid)
        for j in range(W):
            if sel[j] and not scheduled[j]:
                val = int(r_vec[j])
                R[j] = val if val in (1, 2) else 1   # Default to Core mode

        # Prepare action vector for environment step
        # Format: [scheduling_decisions, resource_allocations]
        step_action = np.zeros(W * W + W, dtype=int)
        for j in range(W):
            if sel[j]:
                step_action[j*(W+1)] = 1          # Set diagonal scheduling element
                step_action[W*W + j] = r_vec[j]   # Set resource choice

        # Execute scheduling step in environment and update state
        obs, _, _, _, _ = env.step(step_action)
        
        # Mark selected jobs as scheduled
        for j in range(W):
            if sel[j]:
                scheduled[j] = True

        # Move to next scheduling slot
        row += 1

    # ----------------------
    # 4) Calculate performance metrics
    # ----------------------
    # Calculate Relative Work Factor (RWF) - key performance metric
    # RWF measures performance improvement over traditional time-sharing
    
    # Get baseline solo execution times for each job
    solo_times = [df.iloc[0]["solo_time"] for df in dfs]
    total_solo = sum(solo_times)  # Total time if all jobs run sequentially
    
    # Calculate actual co-run execution times based on scheduling decisions
    corun_times = [df["duration_time"].iloc[:env.indices[i]].sum()/1e6 for i, df in enumerate(dfs)]
    total_corun = max(corun_times)  # Makespan (time when last job completes)
    
    # Calculate RWF: percentage improvement over time-sharing baseline
    rwf = ((total_solo / (total_corun + 1e-6)) - 1) * 100

    # ----------------------
    # 5) Display comprehensive results
    # ----------------------

    #print(f"\n🧾 Final Reward (RWF): {rwf:.2f}% improvement over time-sharing")

    print("\n🧩 Final Co-Scheduling Matrices:")
    print("S: (Rows = time slots, Columns = jobs, 1 = scheduled)")
    print(S)
    print("R: (Resource allocation vector, 1 = Core mode, 2 = Thread mode)")
    print(R)   # length W vector, values in {0,1,2}

    # Display human-readable scheduling groups for each time slot
    print("\n📋 Co-Scheduling Groups:")
    for slot in range(W):
        jobs_in_slot = list(np.where(S[slot] == 1)[0])  # Find jobs scheduled in this slot
        if not jobs_in_slot:
            continue
        # Map resource allocation codes to human-readable modes
        modes = ["Core" if R[j] == 1 else "Thread" for j in jobs_in_slot]
        print(f"  ▪️ Slot {slot}: Jobs {jobs_in_slot} → Modes {modes}")

    print("==============================================\n")
