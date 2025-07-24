from multi_env import MultiJobSchedulingEnv
from stable_baselines3 import PPO
import pandas as pd
import glob

# Load all 10 benchmark CSVs
csv_paths = glob.glob("/home/rishabh2025/profiler/logs/*/*_dataset.csv")
df_pool = [pd.read_csv(p) for p in csv_paths]

# Let env sample 6 jobs randomly each episode
env = MultiJobSchedulingEnv(df_pool, num_jobs=6, use_all_actions=False)
env.shuffle_on_reset = True

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_scheduler_random_mixes")