#!/usr/bin/env python3
"""
Multi-Job Scheduling RL Training Script

This script trains a PPO (Proximal Policy Optimization) agent to make optimal
co-scheduling decisions for multiple HPC jobs. The training uses a curriculum
learning approach with multiple random job mixes to ensure the agent learns
robust scheduling policies across diverse workload combinations.

Key Features:
- Curriculum learning with random job mix sampling
- PPO agent trained on multi-job co-scheduling environment
- Progressive training across 15 different job combinations
- Model persistence for deployment and evaluation

Training Strategy:
- Each mix contains 6 randomly selected HPC benchmarks
- Agent learns to optimize co-scheduling decisions and resource allocation
- Training progresses through diverse workload characteristics
- Final model generalizes across different job combinations

The trained model can be used with sim_multi.py and sim_custom.py for
scheduling evaluation and real-world deployment scenarios.
"""

import os
import random
import numpy as np
import pandas as pd
from multi_env import MultiJobSchedulingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ----------------------
# Configuration and Data Setup
# ----------------------

# Path where all benchmark performance datasets are stored
DATA_DIR = "/home/rishabh2025/profiler/logs"

# List of available HPC benchmarks for training
# Each benchmark represents a different computational workload pattern
BENCHMARKS = [
    "AMG",         # Algebraic MultiGrid - Linear algebra/sparse matrices
    "XSBench",     # Monte Carlo neutron transport simulation
    "minTally",    # Simplified Monte Carlo particle transport
    "simplemoc",   # Method of Characteristics transport solver
    "hpccg",       # High Performance Conjugate Gradient solver
    "CoHMM",       # Codesign Hydrodynamics Miniapp
    "CoMD",        # Codesign Molecular Dynamics
    "miniMD",      # Molecular dynamics simulation miniapp
    "miniQMC",     # Quantum Monte Carlo simulation
    "Quicksilver"  # Monte Carlo transport simulation
]

# Load all benchmark datasets into memory for efficient access during training
# Each dataset contains performance counters collected under different execution scenarios
benchmark_data = {
    name: pd.read_csv(os.path.join(DATA_DIR, name, f"{name}_dataset.csv"))
    for name in BENCHMARKS
}

# ----------------------
# Training Hyperparameters
# ----------------------

NUM_MIXES = 15             # Number of random job mix combinations for curriculum learning
JOBS_PER_MIX = 6           # Number of jobs per scheduling episode (window size W)
TIMESTEPS_PER_MIX = 5000   # Training timesteps per job mix (learning iterations)

# ----------------------
# Curriculum Learning Training Loop
# ----------------------

# Initialize model variable (will be created on first iteration)
model = None

# Progressive training across diverse job combinations
# Each iteration exposes the agent to different workload characteristics
for mix_num in range(NUM_MIXES):
    
    # ----------------------
    # Job Mix Selection and Environment Setup
    # ----------------------
    
    # Randomly sample jobs for this training iteration
    # This ensures the agent learns robust policies across workload diversity
    selected = random.sample(BENCHMARKS, JOBS_PER_MIX)
    
    # Create DataFrame pool for the selected job combination
    df_pool = [benchmark_data[name] for name in selected]

    print(f"\n🔁 Training on Mix {mix_num + 1}: {selected}")
    
    # Initialize multi-job scheduling environment for this mix
    env = MultiJobSchedulingEnv(df_pool, 
                               num_jobs=JOBS_PER_MIX,    # 6 jobs per episode
                               use_all_actions=False)    # Use simplified action space

    # ----------------------
    # Model Initialization and Environment Validation
    # ----------------------
    
    if model is None:
        # First iteration: create new PPO model and validate environment
        obs, _ = env.reset()
        
        # Validate environment interface compliance with Gymnasium standards
        check_env(env, warn=True)  # Optional: validate interface
        
        # Initialize PPO agent with Multi-Layer Perceptron policy
        # MlpPolicy: Neural network for both value function and policy
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        # Subsequent iterations: reuse existing model with new environment
        env.reset()
        
        # Update the model's environment to the new job mix
        # This allows continuous learning across different job combinations
        model.set_env(env)

    # ----------------------
    # Training Phase
    # ----------------------
    
    # Train the agent on the current job mix
    # The agent learns to optimize co-scheduling decisions for this specific combination
    model.learn(total_timesteps=TIMESTEPS_PER_MIX)

# ----------------------
# Model Persistence
# ----------------------

# Save the trained model for deployment and evaluation
model.save("ppo_rwi_scheduler")
print("\n✅ Training completed and model saved as 'ppo_rwi_scheduler.zip'")

"""
Training Summary:
- Curriculum Learning: 15 random job combinations for robust policy learning
- Job Diversity: 10 different HPC benchmarks covering various computational patterns
- Training Volume: 75,000 total timesteps (15 mixes × 5,000 timesteps each)
- Model Architecture: PPO with MLP policy network
- Environment: Multi-job co-scheduling with RWI-based rewards

The trained model learns to:
1. Identify optimal job groupings for co-scheduling
2. Allocate appropriate resources (Core vs Thread mode)
3. Maximize system throughput using Relative Work Index (RWI)
4. Handle diverse workload combinations and resource constraints

Usage:
- Use sim_multi.py to evaluate the trained model on predefined job mixes
- Use sim_custom.py to test custom job combinations interactively
- The model file 'ppo_rwi_scheduler.zip' contains the learned scheduling policy
"""
