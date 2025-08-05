#!/usr/bin/env python3
"""
Multi-Job Co-Scheduling Environment for Reinforcement Learning

This module implements a Gymnasium-compatible environment for training RL agents
to make optimal co-scheduling decisions for multiple HPC jobs simultaneously.
The environment manages job selection, resource allocation, and co-scheduling
groups to maximize overall system throughput.

Key Features:
- Multi-dimensional observation space (num_jobs × 12 performance counters)
- Complex action space for job scheduling matrix and resource allocation
- Group-based reward calculation using connected components
- Support for concurrent job execution with configurable limits (Cmax)
- RWI-based performance optimization across job groups

The environment uses graph theory (connected components) to identify co-scheduling
groups and calculates rewards based on relative performance improvements within
each group compared to solo execution baselines.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiJobSchedulingEnv(gym.Env):
    """
    Multi-Job Co-Scheduling Environment for Reinforcement Learning
    
    This environment simulates co-scheduling decisions for multiple HPC jobs
    running concurrently. The agent learns to create optimal job groupings
    and resource allocations to maximize overall system performance.
    
    Observation Space:
        Box(num_jobs × 12,) - Concatenated performance counters for all jobs:
        - Each job contributes 12 performance metrics
        - Total observation dimension: num_jobs × 12
        - Includes CPU cycles, instructions, cache metrics, etc.
    
    Action Space:
        MultiDiscrete - Two-part action structure:
        - Scheduling Matrix: num_jobs² binary decisions (job co-scheduling)
        - Resource Vector: num_jobs decisions (0=NoRun, 1=Core, 2=Thread)
        - Total action dimension: num_jobs² + num_jobs
    
    Co-scheduling Logic:
        - Uses graph connectivity to identify job groups
        - Connected components in scheduling matrix form co-scheduling groups
        - Each group is subject to concurrency limit (Cmax)
        - Groups execute with shared resource allocation strategies
    
    Reward Calculation:
        - Group-based RWI computation
        - Compares co-run performance to solo execution baselines
        - Incorporates IPC, duration, and cache performance metrics
        - Higher rewards for more efficient resource utilization
    """
    
    def __init__(self, benchmark_pool, num_jobs=6, Cmax=4, use_all_actions=False, shuffle_on_reset=True):
        """
        Initialize the multi-job scheduling environment
        
        Args:
            benchmark_pool (list): List of pandas DataFrames, each containing
                performance data for a different HPC benchmark
            num_jobs (int): Number of jobs to schedule simultaneously (default: 6)
            Cmax (int): Maximum number of jobs that can run concurrently in a group (default: 4)
            use_all_actions (bool): Whether to use extended action space (default: False)
            shuffle_on_reset (bool): Whether to randomly sample jobs on reset (default: True)
        """
        super(MultiJobSchedulingEnv, self).__init__()
        
        # Validate that we have enough benchmarks in the pool
        assert len(benchmark_pool) >= num_jobs, f"Pool has {len(benchmark_pool)} jobs, but need at least {num_jobs}"

        # Store environment configuration
        self.benchmark_pool = benchmark_pool    # List of performance DataFrames
        self.num_jobs = num_jobs               # Number of jobs to schedule per episode
        self.use_all_actions = use_all_actions # Extended action space flag
        self.shuffle_on_reset = shuffle_on_reset # Random job sampling flag
        self.Cmax = Cmax                       # Maximum concurrent jobs per group
        
        # Episode-specific state variables (reset each episode)
        self.jobs = []              # Currently selected job DataFrames
        self.indices = []           # Current row index for each job's execution
        self.max_durations = []     # Maximum duration for each job (normalization)
        self.max_llc_misses = []    # Maximum LLC misses for each job (normalization)

        # Action space configuration
        self.num_actions = 3 if use_all_actions else 2
        
        # Define action space structure:
        # - First num_jobs² elements: Scheduling matrix S (binary decisions)
        # - Next num_jobs elements: Resource allocation vector R (0=NoRun, 1=Core, 2=Thread)
        self.action_space = spaces.MultiDiscrete([2] * (self.num_jobs ** 2) + [3] * self.num_jobs)
        
        # Define observation space: concatenated performance counters for all jobs
        # Shape: (num_jobs × 12,) where each job contributes 12 performance metrics
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_jobs * 12,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state for a new episode
        
        Args:
            seed (int, optional): Random seed for reproducible job selection
            options (dict, optional): Additional reset options
            
        Returns:
            tuple: (initial_observation, info_dict)
                - initial_observation: Concatenated performance counters for all jobs
                - info_dict: Empty dictionary (required by Gymnasium interface)
        """
        super().reset(seed=seed)
        
        # Set random seed for reproducible experiments
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Select jobs for this episode
        if self.shuffle_on_reset:
            # Randomly sample num_jobs from the benchmark pool
            selected_indices = self.np_random.choice(len(self.benchmark_pool), size=self.num_jobs, replace=False)
            self.jobs = [self.benchmark_pool[i] for i in selected_indices]
        else:
            # Use first num_jobs from the pool (deterministic selection)
            self.jobs = self.benchmark_pool[:self.num_jobs]

        # Initialize execution state for each job
        self.indices = [0] * self.num_jobs  # Start all jobs at their first data row
        
        # Precompute normalization factors for reward calculation
        self.max_durations = [df["duration_time"].max() for df in self.jobs]
        self.max_llc_misses = [df["LLC-load-misses"].max() for df in self.jobs]

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construct observation vector from current job states
        
        The observation concatenates performance counters from all jobs,
        creating a unified state representation for the multi-job environment.
        
        Returns:
            np.ndarray: Concatenated performance counters, shape (num_jobs × 12,)
                - Each job contributes 12 performance metrics
                - Jobs that have completed contribute zero vectors
                - Metrics include: CPU cycles, instructions, cache misses, etc.
        """
        obs = []
        
        # Collect performance counters from each job
        for i in range(self.num_jobs):
            idx = self.indices[i]  # Current execution step for job i
            
            if idx < len(self.jobs[i]):
                # Job is still running: extract current performance counters
                row = self.jobs[i].iloc[idx]
                features = row[1:13].values.astype(np.float32)  # Columns 1-13: 12 perf counters
            else:
                # Job has completed: use zero padding
                features = np.zeros(12, dtype=np.float32)
                
            obs.append(features)

        # Concatenate all job features into single observation vector
        return np.concatenate(obs, dtype=np.float32)  # Final shape: (num_jobs * 12,)

    def step(self, action):
        """
        Execute one scheduling step in the multi-job environment
        
        Args:
            action (list/np.ndarray): Combined scheduling and resource allocation decisions
                - First num_jobs² elements: Flattened scheduling matrix S
                - Next num_jobs elements: Resource allocation vector R
                
        Returns:
            tuple: (next_observation, reward, terminated, truncated, info)
                - next_observation: Updated performance counters for all jobs
                - reward: Sum of group-based RWI rewards
                - terminated: True if any job completes
                - truncated: False (not used)
                - info: Empty dictionary
        """
        # Ensure action is in list format for processing
        if isinstance(action, np.ndarray):
            action = action.tolist()

        W = self.num_jobs
        
        # ----------------------
        # Parse Action Components
        # ----------------------
        
        # Extract scheduling matrix (flattened) and resource allocation vector
        s_flat = action[:W * W]    # Scheduling decisions: S[i,j] = 1 if jobs i,j co-scheduled
        r_vec = action[W * W:]     # Resource allocation: R[i] ∈ {0=NoRun, 1=Core, 2=Thread}

        # Reconstruct matrices from flattened action
        S = np.array(s_flat).reshape((W, W))     # Scheduling matrix: W×W
        R = np.array(r_vec)                      # Resource vector: length W

        # ----------------------
        # Identify Co-scheduling Groups using Graph Theory
        # ----------------------
        
        # Use connected components to find job groups that should run together
        # Jobs are connected if S[i,j] = 1, forming co-scheduling groups
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        G = csr_matrix(S)  # Convert scheduling matrix to sparse graph
        n_components, labels = connected_components(csgraph=G, directed=False, return_labels=True)

        rewards = []

        # ----------------------
        # Process Each Co-scheduling Group
        # ----------------------
        
        for group_id in range(n_components):
            # Find all jobs assigned to this group with valid resource allocation
            group_jobs = [i for i in range(W) if labels[i] == group_id and R[i] > 0]
            
            if not group_jobs:
                continue  # Skip empty groups
                
            # Enforce concurrency limit (Cmax)
            if len(group_jobs) > self.Cmax:
                group_jobs = group_jobs[:self.Cmax]  # Truncate to Cmax jobs

            # ----------------------
            # Group-wise Performance Baseline Calculation
            # ----------------------
            
            # Calculate group averages for normalization and comparison
            # These serve as baselines for relative performance measurement
            
            # Average solo execution time across group jobs
            mean_solo = np.mean([
                self.jobs[i].iloc[self.indices[i]].get("solo_time", 1.0)
                for i in group_jobs
                if self.indices[i] < len(self.jobs[i])
            ]) + 1e-6  # Add epsilon to prevent division by zero

            # Average Instructions Per Cycle (IPC) across group jobs
            mean_ipc = np.mean([
                self.jobs[i].iloc[self.indices[i]]["instructions"] / 
                (self.jobs[i].iloc[self.indices[i]]["cpu-cycles"] + 1e-6)
                for i in group_jobs
                if self.indices[i] < len(self.jobs[i])
            ]) + 1e-6

            # Average Last Level Cache (LLC) miss rate across group jobs
            mean_l3 = np.mean([
                self.jobs[i].iloc[self.indices[i]]["LLC-load-misses"]
                for i in group_jobs
                if self.indices[i] < len(self.jobs[i])
            ]) + 1e-6

            # ----------------------
            # Individual Job Reward Calculation within Group
            # ----------------------
            
            for i in group_jobs:
                # Skip jobs that have already completed
                if self.indices[i] >= len(self.jobs[i]):
                    continue

                # Get current performance data for job i
                row = self.jobs[i].iloc[self.indices[i]]
                alloc_type = R[i]  # Resource allocation: 1=Core, 2=Thread

                # ----------------------
                # Resource Allocation Effectiveness
                # ----------------------
                
                # Map allocation type to effectiveness ratio
                core_alloc_ratio = 1.0 if alloc_type == 1 else 0.5  # Core more effective than Thread

                # ----------------------
                # Performance Metrics Calculation
                # ----------------------
                
                # Duration performance: compare to group baseline
                solo_time = row.get("solo_time", 1.0)
                duration_ratio = solo_time / mean_solo

                # IPC performance: compare to group baseline
                ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)
                scale_factor_ratio = ipc / mean_ipc

                # Cache performance: compare to group baseline
                l3 = row["LLC-load-misses"]
                l3_cache_miss_ratio = l3 / mean_l3

                # ----------------------
                # RWI Calculation
                # ----------------------
                
                # Combine metrics into single reward value
                # Higher values indicate better performance relative to group baseline
                rwi = core_alloc_ratio * (scale_factor_ratio**2 + duration_ratio**2) + l3_cache_miss_ratio**2
                rewards.append(rwi)

                # Advance this job to its next execution step
                self.indices[i] += 1

        # ----------------------
        # Environment State Management
        # ----------------------

        # Total episode reward is sum of all individual job rewards
        total_reward = sum(rewards)

        # Episode terminates when any job completes its execution
        terminated = any(self.indices[i] >= len(self.jobs[i]) for i in range(self.num_jobs))
        truncated = False  # Not used in this environment

        # Prepare next observation (or zero vector if terminated)
        obs = self._get_obs() if not terminated else np.zeros(self.num_jobs * 12, dtype=np.float32)
        
        return obs, total_reward, terminated, truncated, {}

