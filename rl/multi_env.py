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

    def step(self, action):
        if isinstance(action, (np.ndarray, tuple)):
            action = list(action)
        elif not isinstance(action, list):
            action = [action] * self.num_jobs  # repeat if scalar  
        rewards = []

        for i in range(self.num_jobs):
            row = self.jobs[i].iloc[self.indices[i]]
            act = action[i]

            # Core allocation ratio (1.0 = compact, 0.5 = thread interleaving)
            core_alloc_ratio = 1.0 if act == 0 else 0.5

            # Duration ratio = solo_time / mean_solo_time
            solo_time = row.get("solo_time", 1.0)
            mean_solo = np.mean([job.iloc[self.indices[i]].get("solo_time", 1.0) for job in self.jobs])
            duration_ratio = solo_time / (mean_solo + 1e-6)

            # IPC = instructions / cpu-cycles
            ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)

            # Scale factor = single-core IPC / multi-core IPC (approximate with IPC for now)
            scale_factor_ratio = ipc / (np.mean([
                job.iloc[self.indices[i]]["instructions"] / (job.iloc[self.indices[i]]["cpu-cycles"] + 1e-6)
                for job in self.jobs
            ]) + 1e-6)

            # L3 cache misses ratio = this_job / mean_of_all
            l3 = row["LLC-load-misses"]
            mean_l3 = np.mean([job.iloc[self.indices[i]]["LLC-load-misses"] for job in self.jobs])
            l3_cache_misses_ratio = l3 / (mean_l3 + 1e-6)

            # Final reward (rwi) — simplified as per paper
            rwi = core_alloc_ratio * (scale_factor_ratio ** 2 + duration_ratio ** 2) + l3_cache_misses_ratio ** 2
            rewards.append(rwi)

        total_reward = sum(rewards)

        # Advance all job time steps
        self.indices = [idx + 1 for idx in self.indices]
        terminated = any(self.indices[i] >= len(self.jobs[i]) for i in range(self.num_jobs))
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.num_jobs * 12, dtype=np.float32)
        return obs, total_reward, terminated, truncated, {}
