import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import glob

# Merge all benchmark CSVs into one DataFrame
csv_paths = glob.glob("/home/rishabh2025/profiler/logs/*/*_dataset.csv")
df_list = [pd.read_csv(path) for path in csv_paths]
merged_df = pd.concat(df_list, ignore_index=True)
merged_df = merged_df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

# Initialize environment with full data
env = SchedulingEnv(merged_df)
check_env(env, warn=True)

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)  # Increase if needed
model.save("ppo_scheduler_all_benchmarks")
