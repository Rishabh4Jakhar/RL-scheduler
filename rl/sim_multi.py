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
model = PPO.load("ppo_scheduler_random_mixes")

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

    avg_reward = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n✅ Total RWI Reward for Mix {mix_index + 1}: {total_reward:.4f}")
    print(f"✅ Average RWI per step: {avg_reward:.4f}")
    print("==============================================\n")
