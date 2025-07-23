import pandas as pd
from env import SchedulingEnv
from stable_baselines3 import PPO

df = pd.read_csv("profiler/profiler/logs/AMG/AMG_dataset.csv")
env = SchedulingEnv(df)

model = PPO.load("ppo_scheduler")

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print("Total IPC (reward):", total_reward)
