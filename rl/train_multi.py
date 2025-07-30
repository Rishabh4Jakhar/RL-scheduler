import os
import random
import pandas as pd
from multi_env import MultiJobSchedulingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Path where all 10 benchmark datasets are stored
DATA_DIR = "/home/rishabh2025/profiler/logs"

BENCHMARKS = [
    "AMG", "XSBench", "minTally", "simplemoc", "hpccg",
    "CoHMM", "CoMD", "miniMD", "miniQMC", "Quicksilver"
]

# Load all datasets into a dict
benchmark_data = {
    name: pd.read_csv(os.path.join(DATA_DIR, name, f"{name}_dataset.csv"))
    for name in BENCHMARKS
}

# Configuration
NUM_MIXES = 15      # Number of random job mixes
JOBS_PER_MIX = 6      # Window size W
TIMESTEPS_PER_MIX = 5000  # Training steps per mix

# Create a training loop across random job mixes
model = None

for mix_num in range(NUM_MIXES):
    selected = random.sample(BENCHMARKS, JOBS_PER_MIX)
    df_pool = [benchmark_data[name] for name in selected]

    print(f"\n🔁 Training on Mix {mix_num + 1}: {selected}")
    env = MultiJobSchedulingEnv(df_pool, num_jobs=JOBS_PER_MIX, use_all_actions=False)
    
    if model is None:
        obs, _ = env.reset()
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        env.reset()
        model.set_env(env)

    model.learn(total_timesteps=TIMESTEPS_PER_MIX)

# Save model after all mixes
model.save("ppo_rwi_scheduler")
print("\n✅ Training completed and model saved as 'ppo_rwi_scheduler.zip'")
    