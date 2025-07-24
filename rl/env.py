import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SchedulingEnv(gym.Env):
    def __init__(self, data_df):
        super(SchedulingEnv, self).__init__()
        self.data = data_df
        self.curr_index = 0
        # State: 12 perf counters
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)
        # Action: 0 = core affinity, 1 = thread interleaving, 2 = numa distribution
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.curr_index = 0

        # Precompute max values for normalization
        self.max_duration = self.data["duration_time"].max()
        self.max_llc_misses = self.data["LLC-load-misses"].max()

        state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        return state, {}

    def step(self, action):
        row = self.data.iloc[self.curr_index]

        # Action-based weight mappings
        core_alloc_ratio = 1.0 if action == 0 else 0.5
        cache_alloc_ratio = 1.0 if action == 1 else 0.5
        bw_alloc_ratio = 1.0 if action == 2 else 0.5

        # Reward components
        duration_ratio = row["duration_time"] / (self.max_duration + 1e-6)
        ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)
        scale_factor_ratio = ipc  # Approximate
        l3_misses_ratio = row["LLC-load-misses"] / (self.max_llc_misses + 1e-6)

        # Compute reward
        rwi = core_alloc_ratio * (scale_factor_ratio ** 2 + duration_ratio ** 2)

        reward = rwi

        self.curr_index += 1
        terminated = self.curr_index >= len(self.data)
        truncated = False

        if not terminated:
            next_state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        else:
            next_state = np.zeros(12, dtype=np.float32)

        return next_state, reward, terminated, truncated, {}
