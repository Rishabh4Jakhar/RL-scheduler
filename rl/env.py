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
        state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        return state, {}

    def step(self, action):
        row = self.data.iloc[self.curr_index]
        ipc = row['instructions']/(row['cpu-cycles'] + 1e-6)
        reward = ipc  # Optional: use paper’s reward formula instead

        self.curr_index += 1
        terminated = self.curr_index >= len(self.data)
        truncated = False

        if not terminated:
            next_state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        else:
            next_state = np.zeros(12, dtype=np.float32)

        return next_state, reward, terminated, truncated, {}
