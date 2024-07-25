import argparse
import os
import sys
import torch
from ppo.envs import make_vec_envs
from ppo.utils import get_render_func,get_vec_normalize

sys.path.append('ppo')
parser=argparse.ArgumentParser(description='Run DRL')
parser.add_argument('--seed',type=int,default=1,help='random seed')
parser.add_argument('--log-interval',type=int,default=10,help='log interval, 1 log per n updates')
parser.add_argument('--env-name',default='QuadEnv-v0',help="Enter the env you want to work on")
parser.add_argument("--load-dir",default="",help="to save agent logs")#add in the default
parser.add_argument("--det",action='store-true',default=True,help="to use deterministic property or nah")
args=parser.parse_args()
args.not_det=not args.det

env=make_vec_envs(args.env_name,args.seed+1000,1,None,None,device='cpu',allow_early_resets=False)
render_func=get_render_func(env)

actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location=lambda storage, loc: storage)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

t_max = 10
t_ = 0
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)
    obs, reward, done, _ = env.step(action)
    masks.fill_(0.0 if done else 1.0)
    if done:
        t_ += 1
        env.reset()
    if render_func is not None:
        render_func('human')
    if t_ > t_max:
        break
