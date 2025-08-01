import os
import sys
import subprocess
import pandas as pd
import numpy as np
from multi_env import MultiJobSchedulingEnv
from stable_baselines3 import PPO

# === Step 1: Parse arguments ===
W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
Cmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4

logs_base = "/home/rishabh2025/profiler/logs"
dataset_suffix = "_dataset.csv"

jobs = []
job_cmds = {}

print(f"\nEnter {W} job names. You can also provide custom jobs like:")
print("  XSBench_custom1 --cmd ../../benchmarks/XSBench/openmp-threading/XSBench -G nuclide -g 7000 -l 200 -p 350000 -t 12")
print("If no --cmd is given, default benchmark logs are used (like 'AMG')")

# === Step 2: Collect job definitions ===
for i in range(W):
    line = input(f"\n▶️ Enter job {i+1}: ").strip()
    if "--cmd" in line:
        parts = line.split("--cmd", maxsplit=1)
        job_name = parts[0].strip()
        job_cmd = parts[1].strip()
        job_cmds[job_name] = job_cmd
    else:
        job_name = line
    jobs.append(job_name)

# === Step 3: Load logs or generate them ===
dfs = []
for job in jobs:
    dataset_path = os.path.join(logs_base, job, f"{job}{dataset_suffix}")
    if not os.path.exists(dataset_path):
        print(f"[!] Logs not found for {job}. Generating...")
        os.makedirs(os.path.join(logs_base, job), exist_ok=True)

        if job in job_cmds:
            run_cmd = job_cmds[job]
            full_cmd = f"""
            perf stat -e duration_time,task-clock,context-switches,cpu-cycles,instructions,LLC-load-misses -I 50 -a -x, -o {logs_base}/{job}/group_A.csv -- bash -c "{run_cmd}"
            """
            print(f"⏳ Running custom command for {job}...")
            subprocess.run(full_cmd, shell=True, executable="/bin/bash")

            subprocess.run(f"python3 scripts/merge.py {job}", shell=True)

        else:
            print(f"⏳ Using collect_dataset.sh for {job}...")
            subprocess.run(f"bash scripts/collect_dataset.sh {job}", shell=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset still not found for {job}")
    dfs.append(pd.read_csv(dataset_path))

# === Step 4: Run simulation ===
env = MultiJobSchedulingEnv(dfs, num_jobs=W, Cmax=Cmax, use_all_actions=False, shuffle_on_reset=False)
model = PPO.load("ppo_rwi_scheduler")
obs, _ = env.reset()

S = np.zeros((W, W), dtype=int)
R = np.zeros(W, dtype=int)
scheduled = [False] * W
row = 0

while row < W and not all(scheduled):
    action, _ = model.predict(obs)
    s_flat = action[:W*W]
    r_vec = action[W*W:W*W+W]
    sel = np.array(s_flat).reshape(W, W).sum(axis=0)
    sel = (sel > 0).astype(int)
    for j in range(W):
        if scheduled[j]:
            sel[j] = 0
    if sel.sum() > Cmax:
        order = np.argsort(-np.array(r_vec))
        keep = order[:Cmax]
        mask = np.zeros(W, dtype=int)
        mask[keep] = 1
        sel = sel * mask

    S[row, :] = sel
    for j in range(W):
        if sel[j]:
            R[j] = int(r_vec[j]) + 1  # 1=Core, 2=Thread

    step_action = np.zeros(W * W + W, dtype=int)
    for j in range(W):
        if sel[j]:
            step_action[j * (W + 1)] = 1
            step_action[W * W + j] = r_vec[j]
    obs, _, _, _, _ = env.step(step_action)

    for j in range(W):
        if sel[j]:
            scheduled[j] = True

    row += 1

# === Step 5: Output Results ===
print(f"\n🧩 Final Co-Scheduling Matrices:")
print("S:")
print(S)
print("R:")
print(R)

print("\n📋 Co-Scheduling Groups:")
for i in range(W):
    jobs_in_group = list(np.where(S[i] == 1)[0])
    if not jobs_in_group:
        continue
    modes = ["Core" if R[j] == 1 else "Thread" for j in jobs_in_group]
    job_names = [jobs[j] for j in jobs_in_group]
    print(f"  ▪️ Slot {i}: {job_names} → Modes {modes}")
