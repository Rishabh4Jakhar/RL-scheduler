import pandas as pd
from stable_baselines3 import PPO
from multi_env import MultiJobSchedulingEnv

# === Define test job mixes (6 jobs each) ===
job_mixes = [
    ["AMG", "XSBench", "minTally", "simplemoc", "hpccg", "CoHMM"],
    ["miniMD", "miniQMC", "Quicksilver", "CoMD", "XSBench", "AMG"],
    ["simplemoc", "CoMD", "hpccg", "miniQMC", "XSBench", "minTally"],
    ["AMG", "CoHMM", "miniMD", "Quicksilver", "CoMD", "hpccg"],
    ["XSBench", "simplemoc", "CoMD", "Quicksilver", "CoHMM", "miniQMC"],
    ["minTally", "hpccg", "miniMD", "XSBench", "AMG", "Quicksilver"]
]

# === Load trained PPO model ===
model = PPO.load("ppo_rwi_scheduler")

# === Evaluation loop ===
for mix_index, mix in enumerate(job_mixes):
    print(f"\n========== Evaluating Mix {mix_index + 1} ==========")
    print(f"Jobs: {mix}")

    # Load job CSVs
    dfs = []
    for job in mix:
        df = pd.read_csv(f"/home/rishabh2025/profiler/logs/{job}/{job}_dataset.csv")
        dfs.append(df)

    # Create evaluation env with fixed job mix (no shuffling)
    env = MultiJobSchedulingEnv(dfs, num_jobs=6, use_all_actions=False, shuffle_on_reset=False)

    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    print(f"{'Step':<5} | {'Actions':<30} | {'Step Reward':<12} | {'Cumulative RWI'}")

    while not done:
        actions_np, _ = model.predict(obs)
        actions = [int(a) for a in actions_np]  # Convert to native Python ints
        obs, reward, terminated, truncated, _ = env.step(actions)
        total_reward += reward
        step_count += 1

        action_map = {0: "Core", 1: "Thread"}  # Simplified (no NUMA)
        action_str = ", ".join([action_map[a] for a in actions])
        print(f"{step_count:<5} | {action_str:<30} | {reward:<12.4f} | {total_reward:.4f}")

        done = terminated or truncated
    # === Final reward calculation (rwf) ===
    # Total solo time = sum of solo_time for all jobs
    solo_times = [df.iloc[0]["solo_time"] for df in dfs]
    total_solo_time = sum(solo_times)

    # Total co-run time = max duration of any job (in microseconds → seconds)
    co_run_times = []
    for i, df in enumerate(dfs):
        durations = df["duration_time"].iloc[:env.indices[i]]
        co_run_times.append(durations.sum() / 1e6)  # convert from ns to seconds

    total_corun_time = max(co_run_times)
    rwf = ((total_solo_time / (total_corun_time + 1e-6)) - 1) * 100

    print(f"🧾 Final Reward (RWF): {rwf:.2f}% improvement over time-sharing")

    avg_reward = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n✅ Total RWI Reward for Mix {mix_index + 1}: {total_reward:.4f}")
    print(f"✅ Average RWI per step: {avg_reward:.4f}")
    print("==============================================\n")
