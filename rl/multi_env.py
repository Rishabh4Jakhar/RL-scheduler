import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiJobSchedulingEnv(gym.Env):
    def __init__(self, benchmark_pool, num_jobs=6, Cmax=4, use_all_actions=False, shuffle_on_reset=True):
        super(MultiJobSchedulingEnv, self).__init__()
        assert len(benchmark_pool) >= num_jobs, f"Pool has {len(benchmark_pool)} jobs, but need at least {num_jobs}"

        self.benchmark_pool = benchmark_pool  # list of dataframes
        self.num_jobs = num_jobs
        self.use_all_actions = use_all_actions
        self.shuffle_on_reset = shuffle_on_reset
        self.Cmax = Cmax
        self.jobs = []      # actual jobs selected per episode
        self.indices = []   # per-job row index
        self.max_durations = []
        self.max_llc_misses = []

        self.num_actions = 3 if use_all_actions else 2
        #self.action_space = spaces.Dict({
        #    "select": spaces.MultiBinary(self.num_jobs),  # job selection vector
        #    "assign": spaces.MultiDiscrete([2] * self.num_jobs)  # 0: Core, 1: Thread
        #})
        self.action_space = spaces.MultiDiscrete([2] * (2 * self.num_jobs))
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
            idx = self.indices[i]
            if idx < len(self.jobs[i]):
                row = self.jobs[i].iloc[idx]
                features = row[1:13].values.astype(np.float32)  # use the 12 perf counters
            else:
                features = np.zeros(12, dtype=np.float32)  # padding if job has ended
            obs.append(features)

        return np.concatenate(obs, dtype=np.float32)  # shape: (num_jobs * 12,)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()

        half = self.num_jobs
        select_mask = action[:half]
        assign_mask = action[half:]
        # Enforce Cmax constraint
        selected_jobs = [i for i, sel in enumerate(select_mask) if sel == 1]
        if len(selected_jobs) > self.Cmax:
            # Clip or randomly drop extra selections
            selected_jobs = selected_jobs[:self.Cmax]

        rewards = []

        for i in selected_jobs:
            if self.indices[i] >= len(self.jobs[i]):
                continue
            row = self.jobs[i].iloc[self.indices[i]]

            act = assign_mask[i]  # 0 = core, 1 = thread

            # Core allocation ratio (1.0 = compact, 0.5 = thread interleaving)
            core_alloc_ratio = 1.0 if act == 0 else 0.5

            # Duration ratio
            solo_time = row.get("solo_time", 1.0)
            mean_solo = np.mean([
                self.jobs[j].iloc[self.indices[j]]["solo_time"]
                for j in selected_jobs
                if self.indices[j] < len(self.jobs[j])
            ]) or 1.0


            duration_ratio = solo_time / (mean_solo + 1e-6)

            # IPC
            ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)
            scale_factor_ratio = ipc / (
                np.mean([
                    self.jobs[j].iloc[self.indices[j]]["instructions"] / 
                    (self.jobs[j].iloc[self.indices[j]]["cpu-cycles"] + 1e-6)
                    for j in selected_jobs
                    if self.indices[j] < len(self.jobs[j])
                ]) + 1e-6
            )



            # L3 cache ratio
            l3 = row["LLC-load-misses"]
            mean_l3 = np.mean([
                self.jobs[j].iloc[self.indices[j]]["LLC-load-misses"]
                for j in selected_jobs
                if self.indices[j] < len(self.jobs[j])
            ]) or 1.0


            l3_cache_misses_ratio = l3 / (mean_l3 + 1e-6)

            # Final reward
            rwi = core_alloc_ratio * (scale_factor_ratio**2 + duration_ratio**2) + l3_cache_misses_ratio**2
            rewards.append(rwi)

            # Advance this job’s time step
            self.indices[i] += 1

        # Total reward = sum of rewards of selected jobs
        total_reward = sum(rewards)

        # Done if any job finishes (or all jobs for stricter criteria)
        terminated = any(self.indices[i] >= len(self.jobs[i]) for i in range(self.num_jobs))
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.num_jobs * 12, dtype=np.float32)
        return obs, total_reward, terminated, truncated, {}

