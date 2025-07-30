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
        # Total: W×W (for S) + W (for R)
        self.action_space = spaces.MultiDiscrete([2] * (self.num_jobs ** 2) + [3] * self.num_jobs)
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

    W = self.num_jobs
    s_flat = action[:W * W]
    r_vec = action[W * W:]

    # Reconstruct matrices
    S = np.array(s_flat).reshape((W, W))     # S[i][j] = 1 means job i runs with job j
    R = np.array(r_vec)                      # R[i] ∈ {0: NoRun, 1: Core, 2: Thread}

    # Derive groups from S (i.e., connected components)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    G = csr_matrix(S)
    n_components, labels = connected_components(csgraph=G, directed=False, return_labels=True)

    rewards = []

    # Process each group of co-scheduled jobs
    for group_id in range(n_components):
        group_jobs = [i for i in range(W) if labels[i] == group_id and R[i] > 0]
        if not group_jobs:
            continue
        if len(group_jobs) > self.Cmax:
            group_jobs = group_jobs[:self.Cmax]  # Cmax cap

        # Group-wise reward computation
        mean_solo = np.mean([
            self.jobs[i].iloc[self.indices[i]].get("solo_time", 1.0)
            for i in group_jobs
            if self.indices[i] < len(self.jobs[i])
        ]) + 1e-6

        mean_ipc = np.mean([
            self.jobs[i].iloc[self.indices[i]]["instructions"] / 
            (self.jobs[i].iloc[self.indices[i]]["cpu-cycles"] + 1e-6)
            for i in group_jobs
            if self.indices[i] < len(self.jobs[i])
        ]) + 1e-6

        mean_l3 = np.mean([
            self.jobs[i].iloc[self.indices[i]]["LLC-load-misses"]
            for i in group_jobs
            if self.indices[i] < len(self.jobs[i])
        ]) + 1e-6

        for i in group_jobs:
            if self.indices[i] >= len(self.jobs[i]):
                continue

            row = self.jobs[i].iloc[self.indices[i]]
            alloc_type = R[i]  # 1 = Core, 2 = Thread

            core_alloc_ratio = 1.0 if alloc_type == 1 else 0.5  # Thread
            solo_time = row.get("solo_time", 1.0)
            duration_ratio = solo_time / mean_solo

            ipc = row["instructions"] / (row["cpu-cycles"] + 1e-6)
            scale_factor_ratio = ipc / mean_ipc

            l3 = row["LLC-load-misses"]
            l3_cache_miss_ratio = l3 / mean_l3

            rwi = core_alloc_ratio * (scale_factor_ratio**2 + duration_ratio**2) + l3_cache_miss_ratio**2
            rewards.append(rwi)

            self.indices[i] += 1


        # Total reward = sum of rewards of selected jobs
        total_reward = sum(rewards)

        # Done if any job finishes (or all jobs for stricter criteria)
        terminated = any(self.indices[i] >= len(self.jobs[i]) for i in range(self.num_jobs))
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.num_jobs * 12, dtype=np.float32)
        return obs, total_reward, terminated, truncated, {}

