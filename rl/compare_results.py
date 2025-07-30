# compare_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv

# === 1. Define the same 6 test mixes ===
JOB_MIXES = [
    ["AMG", "XSBench", "minTally", "simplemoc", "hpccg", "CoHMM"],
    ["miniMD", "miniQMC", "Quicksilver", "CoMD", "XSBench", "AMG"],
    ["simplemoc", "CoMD", "hpccg", "miniQMC", "XSBench", "minTally"],
    ["AMG", "CoHMM", "miniMD", "Quicksilver", "CoMD", "hpccg"],
    ["XSBench", "simplemoc", "CoMD", "Quicksilver", "CoHMM", "miniQMC"],
    ["minTally", "hpccg", "miniMD", "XSBench", "AMG", "Quicksilver"]
]

# === 2. Load datasets into a dict ===
DATA_DIR = "/home/rishabh2025/profiler/logs"
bench_data = {
    name: pd.read_csv(f"{DATA_DIR}/{name}/{name}_dataset.csv")
    for mix in JOB_MIXES for name in mix
}

# === 3. Load trained RL model ===
model = PPO.load("ppo_rwi_scheduler")

# === 4. Helper functions to compute co-run time ===
def solo_time_total(dfs):
    """Sum of solo_time for each job in seconds."""
    return sum(df.iloc[0]["solo_time"] for df in dfs)

def corruntime_under_policy(dfs, policy_fn, Cmax=4):
    """
    Compute the co-run time (makespan) under a given scheduling policy.
    policy_fn(env, obs) -> action vector.
    """
    env = MultiJobSchedulingEnv(dfs, num_jobs=len(dfs), Cmax=Cmax,
                                use_all_actions=False, shuffle_on_reset=False)
    obs, _ = env.reset()

    # track each job's durations
    durations_ns = [[] for _ in range(len(dfs))]

    done = False
    while not done:
        action = policy_fn(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        # record each job's latest duration_time
        for i, df in enumerate(dfs):
            idx = env.indices[i]-1
            if idx >= 0 and idx < len(df):
                durations_ns[i].append(df.iloc[idx]["duration_time"])
        done = terminated or truncated

    # makespan = longest accumlated time among jobs
    corun_s = max(sum(seq)/1e9 for seq in durations_ns)
    return corun_s

# --- 4a. Baseline policy: Time‑sharing FIFO (no co-run) ---
def baseline_policy(env, obs):
    W = env.num_jobs
    action = np.zeros(W*W + W, dtype=int)
    # schedule exactly one job (next unfinished) at full resource (Core)
    for i in range(W):
        if env.indices[i] < len(env.jobs[i]):
            # pick that job: set S_diag[i]=1, R[i]=1 (Core)
            action[i*(W+1)] = 1
            action[W*W + i] = 1
            break
    return action

# --- 4b. RL-FIFO policy: pick first Cmax unfinished jobs, assignment FIFO ---
def rl_fifo_policy(env, obs):
    W, Cmax = env.num_jobs, env.Cmax
    action = np.zeros(W*W + W, dtype=int)
    # first, pick up to Cmax unfinished in queue order
    unfinished = [i for i in range(W) if env.indices[i] < len(env.jobs[i])]
    selected = unfinished[:Cmax]
    for j in selected:
        action[j*(W+1)] = 1       # S_diag[j] = 1
        action[W*W + j] = 1      # R[j] = Core
    return action

# --- 4c. Our RL policy ---
def rl_policy(env, obs):
    action, _ = model.predict(obs)
    return action

# === 5. Run experiments ===
throughputs = {"Baseline": [], "RL-FIFO": [], "Our RL": []}
labels = [f"Q{idx+1}" for idx in range(len(JOB_MIXES))]

for mix in JOB_MIXES:
    dfs = [bench_data[name] for name in mix]
    solo = solo_time_total(dfs)

    # Baseline
    t_base = corruntime_under_policy(dfs, baseline_policy)
    throughputs["Baseline"].append(solo / t_base)

    # RL-FIFO
    t_fifo = corruntime_under_policy(dfs, rl_fifo_policy)
    throughputs["RL-FIFO"].append(solo / t_fifo)

    # Our RL
    t_rl = corruntime_under_policy(dfs, rl_policy)
    throughputs["Our RL"].append(solo / t_rl)

# === 6. Plot grouped bar chart ===
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, throughputs["Baseline"], width, label="Time Sharing (Baseline)", color="gold")
ax.bar(x,       throughputs["RL-FIFO"], width, label="RL‑FIFO", color="skyblue")
ax.bar(x + width,throughputs["Our RL"],   width, label="Our RL (Cmax=4)", color="navy")

# Horizontal line at y=1 (baseline)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Normalized Throughput")
ax.set_title("Throughput Comparison (W=6, Cmax=4)")
ax.legend(loc='upper left', frameon=True)
ax.set_ylim(0, max(max(v) for v in throughputs.values()) * 1.2)

# instead of plt.show():
output_path = "/home/rishabh2025/profiler/rl/throughput_comparison.png"
plt.tight_layout()
plt.savefig(output_path, dpi=200)
print(f"\n📈 Saved throughput comparison plot to {output_path}")
