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

    def reset(self):
        self.curr_index = 0
        state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        return state

    def step(self, action):
        # Reward: IPC = instructions/cpu-cycles
        row = self.data.iloc[self.curr_index]
        ipc = row['instructions']/(row['cpu-cycles'] + 1e-6)

        reward = ipc # Can be changed to a more complex reward function like the one given in the research paper

        self.curr_index += 1
        done = self.curr_index >= len(self.data)

        if not done:
            next_state = self.data.iloc[self.curr_index, 1:13].values.astype(np.float32)
        else:
            next_state = np.zeros(12, dtype=np.float32)

        return next_state, reward, done, {}
