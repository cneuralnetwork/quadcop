import os
import time
import numpy as np
import gym
import torch
from stable_baselines3 import PPO as pp
from ppo.model import Policy
from ppo.args import get_args
from ppo.utils import cleanup_log_dir,get_vec_normalize
from ppo.envs import make_vec_envs
from ppo.main_algo.ppo import PPO
from ppo.storage import RolloutStorage
from ppo.evaluate import evaluate
from collections import deque

def main():
    args=get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deteministic:
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device=torch.device("cuda:0" if args.cuda else "cpu")

    envs=make_vec_envs(args.env_name,args.seed,args.num_processes,args.gamma,args.log_dir,device,False)
    base_kwargs=dict(recurrent=args.recurrent_polict,hidden_size=args.hidden_size,model="quadrotor")

    _str_dashed_line_sepratr = "-"*46
    print(_str_dashed_line_sepratr)
    print("Model-base param:")
    print(base_kwargs)
    print(print(_str_dashed_line_sepratr))

    actor_critic=Policy(envs.observation_space.shape,envs.action_space,base_kwargs=base_kwargs)
    # (if you wanna use sb3 use this)agent=pp(actor_critic,args.clip_param,args.ppo_epoch,args.num_mini_batch)
    # for custom use PPO() command for custom PPO

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    #loop and run the ppo here

    # print("Steps"+" "*(str_len-len("Steps")) + "|" + f"{j:10d}")
    # print("FPS"+" "*(str_len-len("FPS")) + "|" + f"{int(total_num_steps / (end - start)):10d}")
    # print("#Episodes in batch"+" "*(str_len-len("#Episodes in batch")) + "|" + f"{len(episode_rewards):10d}")
    # print("Mean Reward"+" "*(str_len-len("Mean Reward")) + "|" + f"{np.mean(episode_rewards):10.1f}")
    # print("Median Reward"+" "*(str_len-len("Median Reward")) + "|" + f"{np.median(episode_rewards):10.1f}")
    # print("Minimum Reward"+" "*(str_len-len("Minimum Reward")) + "|" + f"{np.min(episode_rewards):10.1f}")
    # print("Maximum Reward"+" "*(str_len-len("Maximum Reward")) + "|" + f"{np.max(episode_rewards):10.1f}")
    # print("Value Loss"+" "*(str_len-len("Value Loss")) + "|" + f"{value_loss:10.3f}")
    # print("Action Loss"+" "*(str_len-len("Action Loss")) + "|" + f"{action_loss:10.3f}")
    # print("-"*36)
    # if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0: (j loop var)
    #     obs_rms = get_vec_normalize(envs).obs_rms
    #     evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)


if __name__ == "__name__":
    main()
