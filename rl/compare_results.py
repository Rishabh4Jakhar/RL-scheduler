# compare_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv

np.random.seed(0) # For reproducibility

# === 1. Define 12 six-job mixes using your 10 benchmarks ===
BENCHS = ["AMG", "simplemoc", "Quicksilver", "minTally", "miniQMC",
          "miniMD", "hpccg", "XSBench", "CoHMM", "CoMD", "XSBench_custom1", "simplemoc_custom1", "miniQMC_custom1", "Quicksilver_custom1", "miniMD_custom1", "CoMD_custom1"]

JOB_MIXES = [
    ["AMG", "hpccg", "miniMD", "simplemoc", "XSBench_custom1", "CoHMM"], 
    ["simplemoc_custom1", "Quicksilver", "AMG", "miniMD", "hpccg", "miniQMC"], 
    ["CoMD_custom1", "XSBench", "hpccg", "AMG", "miniQMC", "simplemoc"], 
    ["miniMD_custom1", "hpccg", "Quicksilver", "XSBench", "AMG", "CoHMM"], 
    ["miniQMC_custom1", "simplemoc", "hpccg", "miniMD", "CoMD", "AMG"], 
    ["Quicksilver_custom1", "XSBench", "simplemoc", "AMG", "CoMD", "hpccg"],
    ["AMG", "simplemoc", "Quicksilver", "minTally", "miniMD", "XSBench"],
    ["hpccg", "CoHMM", "CoMD", "AMG", "simplemoc", "miniQMC"],
    ["miniQMC", "hpccg", "minTally", "AMG", "XSBench", "miniMD"],
    ["Quicksilver", "CoMD", "XSBench", "simplemoc", "CoHMM", "hpccg"],
    ["minTally", "AMG", "CoMD", "miniMD", "CoHMM", "XSBench"],
    ["Quicksilver", "miniQMC", "hpccg", "CoHMM", "miniMD", "AMG"],
]

# === 2. Preload all CSVs ===
DATA_DIR = "/home/rishabh2025/profiler/logs"
bench_dfs = {
    name: pd.read_csv(f"{DATA_DIR}/{name}/{name}_dataset.csv")
    for name in BENCHS
}

# === 3. Load your trained PPO model ===
model = PPO.load("ppo_rwi_scheduler")

# === 4. Helpers to compute co-run makespan using 'time' column ===
def corruntime(dfs, policy_fn, Cmax=4):
    env = MultiJobSchedulingEnv(dfs, num_jobs=len(dfs),
                                Cmax=Cmax,
                                use_all_actions=False,
                                shuffle_on_reset=False)
    obs, _ = env.reset(seed=1234) # Reproducible
    done = False
    while not done:
        action = policy_fn(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # each job's makespan = last 'time' sample (sec)
    ms = []
    for i, df in enumerate(dfs):
        idx = min(env.indices[i]-1, len(df)-1)
        ms.append(df.iloc[idx]["time"])
    return max(ms)

# --- Policies ---
def baseline_policy(env, obs):
    W=env.num_jobs
    act = np.zeros(W*W + W, int)
    # one job at a time, Core
    for i in range(W):
        if env.indices[i] < len(env.jobs[i]):
            act[i*(W+1)]   = 1
            act[W*W + i]   = 1
            break
    return act

def rl_fifo_policy(env, obs):
    W, Cmax = env.num_jobs, env.Cmax
    act = np.zeros(W*W + W, int)
    unfinished = [i for i in range(W) if env.indices[i] < len(env.jobs[i])]
    for j in unfinished[:Cmax]:
        act[j*(W+1)]   = 1
        act[W*W + j]   = 1
    return act

def rl_policy(env, obs):
    action, _ = model.predict(obs) # Add deterministic = True for Reproducible
    return action

# === 5. Compute normalized throughputs ===
labels = [f"Q{i+1}" for i in range(len(JOB_MIXES))]
throughputs = {"Baseline": [], "RL-FIFO": [], "Our RL": []}

for mix in JOB_MIXES:
    dfs = [bench_dfs[name] for name in mix]
    solo_total = sum(df.iloc[0]["solo_time"] for df in dfs)  # seconds

    # Baseline = 1.0
    throughputs["Baseline"].append(1.0)

    # RL-FIFO
    t_fifo = corruntime(dfs, rl_fifo_policy)
    throughputs["RL-FIFO"].append(solo_total / (t_fifo*1000))

    # Our RL
    t_rl = corruntime(dfs, rl_policy)
    throughputs["Our RL"].append(solo_total / (t_rl*1000))

# Append Arithmetic Mean 'AM'
for key in throughputs:
    throughputs[key].append(np.mean(throughputs[key]))
labels.append("AM")

# === 6. Plot ===
x = np.arange(len(labels))
w = 0.25
fig, ax = plt.subplots(figsize=(12,6))

ax.bar(x - w, throughputs["Baseline"], w, label="Time Sharing", color="gold")
ax.bar(x,     throughputs["RL-FIFO"],   w, label="RL-FIFO",       color="skyblue")
ax.bar(x + w, throughputs["Our RL"],    w, label="Our RL (Cmax=4)", color="navy")

ax.axhline(1.0, color='red', linestyle='--', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Normalized Throughput")
ax.set_title("Throughput Comparison (W=6, Cmax=4)")
ax.legend(loc='upper left')
ax.set_ylim(0, max(max(v) for v in throughputs.values())*1.2)

plt.tight_layout()
out = "/home/rishabh2025/profiler/rl/throughput_comparison_12.png"
plt.savefig(out, dpi=200)
print(f"Saved plot to {out}")
