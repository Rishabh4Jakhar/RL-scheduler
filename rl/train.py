import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Load and prepare dataset
df = pd.read_csv("data/amg.csv")  # Replace with your merged CSV path

env = SchedulingEnv(df)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

model.save("ppo_scheduler")
