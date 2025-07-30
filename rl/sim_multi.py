import pandas as pd
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

job_mixes = [
    ["AMG", "XSBench", "minTally", "simplemoc", "hpccg", "CoHMM"],
    ["miniMD", "miniQMC", "Quicksilver", "CoMD", "XSBench", "AMG"],
    ["simplemoc", "CoMD", "hpccg", "miniQMC", "XSBench", "minTally"],
    ["AMG", "CoHMM", "miniMD", "Quicksilver", "CoMD", "hpccg"],
    ["XSBench", "simplemoc", "CoMD", "Quicksilver", "CoHMM", "miniQMC"],
    ["minTally", "hpccg", "miniMD", "XSBench", "AMG", "Quicksilver"]
]

model = PPO.load("ppo_rwi_scheduler")

for mix_index, mix in enumerate(job_mixes):
    print(f"\n========== Evaluating Mix {mix_index + 1} ==========")
    print(f"Jobs: {mix}")

    # load data and init env
    dfs = [pd.read_csv(f"/home/rishabh2025/profiler/logs/{job}/{job}_dataset.csv") for job in mix]
    env = MultiJobSchedulingEnv(dfs, num_jobs=6, Cmax=4, use_all_actions=False, shuffle_on_reset=False)
    obs, _ = env.reset()

    W = env.num_jobs
    Cmax = env.Cmax

    # track scheduling
    scheduled = [False] * W

    # S: W×W matrix (slot × job), R: global length-W vector
    S = np.zeros((W, W), dtype=int)
    R = np.zeros(W, dtype=int)

    row = 0
    while row < W and not all(scheduled):
        action, _ = model.predict(obs)

        # unpack
        s_flat = action[: W * W]
        r_vec = action[W * W : W * W + W]

        # candidate select mask = any 1 in column of S_flat
        sel = (np.array(s_flat).reshape(W, W).sum(axis=0) > 0).astype(int)

        # mask out already scheduled
        for j in range(W):
            if scheduled[j]:
                sel[j] = 0

        # enforce Cmax by picking top-Cmax jobs by r_vec
        if sel.sum() > Cmax:
            order = np.argsort(-np.array(r_vec))
            keep = set(order[:Cmax])
            sel = np.array([1 if (sel[j] and j in keep) else 0 for j in range(W)])

        # record this slot in S
        S[row, :] = sel

        # fill R[j] if first time j is scheduled: force 0→1 (Core) so no one is left unassigned
        for j in range(W):
            if sel[j] and not scheduled[j]:
                val = int(r_vec[j])
                R[j] = val if val in (1, 2) else 1   # default 1=Core

        # build a step_action that only advances these jobs
        step_action = np.zeros(W * W + W, dtype=int)
        for j in range(W):
            if sel[j]:
                step_action[j*(W+1)] = 1          # set S_diag[j] = 1
                step_action[W*W + j] = r_vec[j]  # preserve resource choice

        # step env and mark scheduled
        obs, _, _, _, _ = env.step(step_action)
        for j in range(W):
            if sel[j]:
                scheduled[j] = True

        row += 1

    # compute final rwf
    solo_times = [df.iloc[0]["solo_time"] for df in dfs]
    total_solo = sum(solo_times)
    corun_times = [df["duration_time"].iloc[:env.indices[i]].sum()/1e6 for i, df in enumerate(dfs)]
    total_corun = max(corun_times)
    rwf = ((total_solo / (total_corun + 1e-6)) - 1) * 100

    print(f"\n🧾 Final Reward (RWF): {rwf:.2f}% improvement over time-sharing")

    print("\n🧩 Final Co-Scheduling Matrices:")
    print("S:")
    print(S)
    print("R:")
    print(R)   # length W vector, values in {0,1,2}

    # Pretty‑print each scheduling slot (row of S) independently
    print("\n📋 Co-Scheduling Groups:")
    for slot in range(W):
        jobs_in_slot = list(np.where(S[slot] == 1)[0])
        if not jobs_in_slot:
            continue
        modes = ["Core" if R[j] == 1 else "Thread" for j in jobs_in_slot]
        print(f"  ▪️ Slot {slot}: Jobs {jobs_in_slot} → Modes {modes}")

    print("==============================================\n")
