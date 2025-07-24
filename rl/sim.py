import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO

# Predefined 6-job mixes from Table 5 or your own
job_mixes = [
    ["AMG", "XSBench", "minTally", "simplemoc", "hpccg", "CoHMM"],
    ["miniMD", "miniQMC", "Quicksilver", "CoMD", "XSBench", "AMG"],
    # Add more mixes as needed
]

model = PPO.load("ppo_scheduler_all_benchmarks")

print(f"\n{'Mix':<30} {'Job':<15} {'Action':<20} {'RWI (Reward)':<15}")

for i, mix in enumerate(job_mixes):
    print(f"\nEvaluating Mix {i + 1}: {mix}")
    job_rewards = []
    mix_total_reward = 0

    for job in mix:
        path = f"/home/rishabh2025/profiler/logs/{job}/{job}_dataset.csv"
        df = pd.read_csv(path)
        env = SchedulingEnv(df)
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action_np, _ = model.predict(obs)
            action = int(action_np)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        job_rewards.append((job, action, total_reward))
        mix_total_reward += total_reward

    for job, action, reward in job_rewards:
        action_map = {0: "Core Affinity", 1: "Thread Interleaving", 2: "NUMA Distribution"}
        print(f"{'Mix ' + str(i + 1):<30} {job:<15} {action_map[action]:<20} {reward:.4f}")

    print(f"{'':<30} {'TOTAL MIX REWARD':<35} {mix_total_reward:.4f}")
