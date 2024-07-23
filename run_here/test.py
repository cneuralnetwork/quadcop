import gym
from stable_baselines3 import PPO
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import environments

env_name="QuadEnv-v0"
model=PPO.load(f"./policies/{env_name}")
env=gym.make(env_name)
obs=env.reset()

while True:
    act,_states=model.predict(obs)
    obs,reward,done,data=env.step(act)
    env.render()
    if done:
        obs=env.reset()