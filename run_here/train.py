import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import environments

seed=123
env_name="QuadEnv-v0"
policy="" #input here 
vec_env=make_vec_env(env_name,n_envs=8,seed=seed)

model=PPO(policy,vec_env,verbose=1,policy_kwargs=dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[256,256],vf=[256,256])),learning_rate=0.00005,clip_range=0.05,seed=seed,batch_size=256,max_grad_norm=0.2)

model.learn(total_timesteps=20000000)
model.save(f"./policies/{env_name}")
del model
vec_env.close()