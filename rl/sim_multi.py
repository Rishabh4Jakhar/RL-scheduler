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

    dfs = [pd.read_csv(f"/home/rishabh2025/profiler/logs/{job}/{job}_dataset.csv") for job in mix]
    env = MultiJobSchedulingEnv(dfs, num_jobs=6, use_all_actions=False, shuffle_on_reset=False)

    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    S_final = None
    R_final = None

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

        W = env.num_jobs
        s_flat = action[:W * W]
        r_vec = action[W * W:]

        S_final = np.array(s_flat).reshape((W, W))
        R_final = np.array(r_vec)

        done = terminated or truncated

    # Final reward improvement calculation
    solo_times = [df.iloc[0]["solo_time"] for df in dfs]
    total_solo_time = sum(solo_times)
    co_run_times = [df["duration_time"].iloc[:env.indices[i]].sum() / 1e6 for i, df in enumerate(dfs)]
    total_corun_time = max(co_run_times)
    rwf = ((total_solo_time / (total_corun_time + 1e-6)) - 1) * 100

    print(f"\n🧾 Final Reward (RWF): {rwf:.2f}% improvement over time-sharing")
    print("\n🧩 Final Co-Scheduling Matrices:")
    print("S (Selection Matrix):")
    print(S_final.astype(int))
    print("R (Resource Assignment Vector):")
    print(R_final.astype(int))

    print("\n📋 Final Co-Scheduling Decision:")
    graph = csr_matrix(S_final)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    for group_id in range(n_components):
        group_jobs = [i for i, label in enumerate(labels) if label == group_id and R_final[i] > 0]
        if not group_jobs:
            continue
        print(f"  ▪️ Group {group_id}:")
        for i in group_jobs:
            rtype = "Core" if R_final[i] == 1 else "Thread" if R_final[i] == 2 else "Unassigned"
            print(f"     - Job {i} → {rtype}")

    avg_reward = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n✅ Total RWI Reward for Mix {mix_index + 1}: {total_reward:.4f}")
    print(f"✅ Average RWI per step: {avg_reward:.4f}")
    print("==============================================\n")
