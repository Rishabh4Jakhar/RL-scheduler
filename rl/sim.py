import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO

# List of benchmark CSV files (simulate jobs waiting in queue)
benchmark_files = [
    "profiler/logs/AMG/AMG_dataset.csv",
    "profiler/logs/XSBench/XSBench_dataset.csv",
    "profiler/logs/minTally/minTally_dataset.csv",
    "profiler/logs/simplemoc/simplemoc_dataset.csv",
    "profiler/logs/hpccg/hpccg_dataset.csv",
    "profiler/logs/CoHMM/CoHMM_dataset.csv",
    "profiler/logs/CoMD/CoMD_dataset.csv",
    "profiler/logs/miniMD/miniMD_dataset.csv",
    "profiler/logs/miniQMC/miniQMC_dataset.csv",
    "profiler/logs/Quicksilver/Quicksilver_dataset.csv",
    ]

# Load trained RL model
model = PPO.load("ppo_scheduler")

# Simulate job queue
job_states = []
actions = []
rewards = []

print(f"{'Job':<15} {'Action':<10} {'IPC (Reward)':<15}")

for file in benchmark_files:
    df = pd.read_csv(file)
    env = SchedulingEnv(df)

    obs = env.reset()
    done = False
    total_ipc = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_ipc += reward

    job_name = file.split("/")[-1].replace(".csv", "")
    print(f"{job_name:<15} {action:<10} {total_ipc:.3f}")
