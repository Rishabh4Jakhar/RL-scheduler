import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiJobSchedulingEnv(gym.Env):
    def __init__(self, benchmark_pool, num_jobs=6, use_all_actions=False, shuffle_on_reset=True):
        super(MultiJobSchedulingEnv, self).__init__()
        assert len(benchmark_pool) >= num_jobs, f"Pool has {len(benchmark_pool)} jobs, but need at least {num_jobs}"

        self.benchmark_pool = benchmark_pool  # list of dataframes
        self.num_jobs = num_jobs
        self.use_all_actions = use_all_actions
        self.shuffle_on_reset = shuffle_on_reset

        self.jobs = []      # actual jobs selected per episode
        self.indices = []   # per-job row index
        self.max_durations = []
        self.max_llc_misses = []

        self.num_actions = 3 if use_all_actions else 2
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_jobs)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_jobs * 12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Randomly sample `num_jobs` from pool
        if self.shuffle_on_reset:
            selected_indices = self.np_random.choice(len(self.benchmark_pool), size=self.num_jobs, replace=False)
            self.jobs = [self.benchmark_pool[i] for i in selected_indices]
        else:
            self.jobs = self.benchmark_pool[:self.num_jobs]

        self.indices = [0] * self.num_jobs
        self.max_durations = [df["duration_time"].max() for df in self.jobs]
        self.max_llc_misses = [df["LLC-load-misses"].max() for df in self.jobs]

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(self.num_jobs):
            row = self.jobs[i].iloc[self.indices[i]]
            obs.extend(row[1:13].values.astype(np.float32))
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        rewards = []
        for i in range(self.num_jobs):
            row = self.jobs[i].iloc[self.indices[i]]
            action = actions[i]

            # Apply action semantics (core affinity or thread interleaving)
            core_alloc_ratio = 1.0 if action == 0 else 0.5
            duration_ratio = row["duration_time"] / (self.max_durations[i] + 1e-6)
            ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)
            scale_factor_ratio = ipc

            rwi = core_alloc_ratio * (scale_factor_ratio ** 2 + duration_ratio ** 2)
            rewards.append(rwi)

        total_reward = sum(rewards)
        self.indices = [idx + 1 for idx in self.indices]
        terminated = any(self.indices[i] >= len(self.jobs[i]) for i in range(self.num_jobs))
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.num_jobs * 12, dtype=np.float32)
        return obs, total_reward, terminated, truncated, {}
