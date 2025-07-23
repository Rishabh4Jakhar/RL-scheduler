import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO

# List of benchmark CSV files (simulate jobs waiting in queue)
benchmark_files = [
    "/home/rishabh2025/profiler/logs/AMG/AMG_dataset.csv",
    "/home/rishabh2025/profiler/logs/XSBench/XSBench_dataset.csv",
    "/home/rishabh2025/profiler/logs/minTally/minTally_dataset.csv",
    "/home/rishabh2025/profiler/logs/simplemoc/simplemoc_dataset.csv",
    "/home/rishabh2025/profiler/logs/hpccg/hpccg_dataset.csv",
    "/home/rishabh2025/profiler/logs/CoHMM/CoHMM_dataset.csv",
    "/home/rishabh2025/profiler/logs/CoMD/CoMD_dataset.csv",
    "/home/rishabh2025/profiler/logs/miniMD/miniMD_dataset.csv",
    "/home/rishabh2025/profiler/logs/miniQMC/miniQMC_dataset.csv",
    "/home/rishabh2025/profiler/logs/Quicksilver/Quicksilver_dataset.csv",
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
    action_map = {0: "Core Affinity", 1: "Thread Interleaving", 2: "NUMA Distribution"}
    obs, _ = env.reset()
    done = False
    total_ipc = 0
    while True:
        action,_ = model.predict(obs) 
        obs,reward,terminated,truncated, _= env.step(action)
        print(f"Job\t\tAction\tIPC (Reward): {reward:.4f}")
        if terminated or truncated:
            break    
    job_name = file.split("/")[-1].replace(".csv", "")
    print(f"{job_name:<15} {action:<10} {total_ipc:.3f}")
