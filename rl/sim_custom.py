#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv

# ----------------------
# 1) Parse args
# ----------------------
W    = int(sys.argv[1]) if len(sys.argv) > 1 else 6
Cmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4

print(f"\nEnter {W} job names. You can also provide custom jobs like:")
print("  XSBench_custom1 --cmd /full/path/to/XSBench -G nuclide -g 7000 -l 200 -p 350000 -t 12")
print("If no --cmd is given, default benchmark logs are used (e.g. 'AMG')\n")

jobs            = []
custom_commands = {}

for i in range(W):
    line = input(f"▶️ Enter job {i+1}: ").strip()
    if not line:
        print("❌ Empty input, exiting.")
        sys.exit(1)
    if "--cmd" in line:
        name, cmd = line.split("--cmd", 1)
        name = name.strip()
        cmd  = cmd.strip()
        jobs.append(name)
        custom_commands[name] = cmd
    else:
        jobs.append(line)

# ----------------------
# 2) Ensure logs exist
# ----------------------
LOG_BASE = os.path.expanduser("~/profiler/logs")
COLLECT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "scripts", "collect_dataset.sh"))

for job in jobs:
    ds_dir  = os.path.join(LOG_BASE, job)
    ds_file = os.path.join(ds_dir, f"{job}_dataset.csv")
    if os.path.exists(ds_file):
        print(f"✅ Found existing logs for {job}")
        continue

    print(f"[!] Logs not found for {job}, running collect_dataset.sh…")
    os.makedirs(ds_dir, exist_ok=True)

    if job in custom_commands:
        # write a temp custom_commands.txt
        cc_path = os.path.join(os.path.dirname(__file__), "custom_commands.txt")
        with open(cc_path, "a") as f:
            f.write(f"{job}::{custom_commands[job]}\n")

    # invoke the script
    ret = subprocess.call(f"bash {COLLECT} {job}", shell=True)
    if ret != 0 or not os.path.exists(ds_file):
        print(f"❌ Failed to generate logs for {job}")
        sys.exit(1)

    print(f"✅ Logs generated for {job}")

# ----------------------
# 3) Load dataframes
# ----------------------
dfs = []
for job in jobs:
    df = pd.read_csv(os.path.join(LOG_BASE, job, f"{job}_dataset.csv"))
    dfs.append(df)

# ----------------------
# 4) Load RL model and create env
# ----------------------
model = PPO.load(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "ppo_rwi_scheduler.zip")))
env   = MultiJobSchedulingEnv(dfs, num_jobs=W, Cmax=Cmax,
                              use_all_actions=False,
                              shuffle_on_reset=False)

obs, _ = env.reset()

# ----------------------
# 5) Simulate and build S, R
# ----------------------
S         = np.zeros((W, W), dtype=int)
R         = np.zeros(W, dtype=int)
scheduled = [False] * W
row       = 0

while row < W and not all(scheduled):
    action, _ = model.predict(obs, deterministic=True)
    s_flat    = action[: W*W]
    r_vec     = action[W*W : W*W + W]

    sel = s_flat.reshape(W, W).sum(axis=0)
    sel = (sel > 0).astype(int)
    # mask out already scheduled
    for j in range(W):
        if scheduled[j]:
            sel[j] = 0
    # enforce Cmax
    if sel.sum() > Cmax:
        order = np.argsort(-r_vec)
        keep  = order[:Cmax]
        mask  = np.zeros(W, dtype=int)
        mask[keep] = 1
        sel = sel * mask

    S[row, :] = sel
    for j in range(W):
        if sel[j]:
            R[j] = int(r_vec[j]) + 1  # 1 = Core, 2 = Thread

    # advance env
    step_act = np.zeros(W*W + W, dtype=int)
    for j in range(W):
        if sel[j]:
            step_act[j*(W+1)]     = 1
            step_act[W*W + j]     = r_vec[j]
    obs, _, term, trunc, _ = env.step(step_act)

    for j in range(W):
        if sel[j]:
            scheduled[j] = True
    row += 1
    if term or trunc:
        break

# ----------------------
# 6) Print Results
# ----------------------
print("\n🧩 Final Co-Scheduling Matrix S:")
print(S)
print("\n🧩 Final Resource Vector R:")
print(R)

print("\n📋 Co-Scheduling Groups:")
for i in range(W):
    group = np.where(S[i] == 1)[0]
    if len(group) == 0:
        continue
    modes = ["Core" if R[j] == 1 else "Thread" for j in group]
    names = [jobs[j] for j in group]
    print(f"  ▪️ Slot {i}: {names} → {modes}")

print("\n✅ Done.")
