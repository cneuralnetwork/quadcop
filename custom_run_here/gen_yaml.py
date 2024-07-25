import os
import argparse
import yaml

parser=argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seeds', type=int,default=5,help="Enter number of random seed to generate")
parser.add_argument('--env',default="QuadEnv-v0",help='Enter name of environment to be used')
parser.add_argument('--yaml',default="run_all",help="Enter the name of the yaml file to be added")
parser.add_argument('--type-of-training',default="default",help='Enter the type of training')
parser.add_argument('--dev-weight-freeze',default=False,action='store_true',help="It will freeze the weights in the dev policy.")
parser.add_argument('--python-path',default="python",help="Add in the path of the place where your python interpreter is")
args=parser.parse_args()

_this_file_dir = os.path.dirname(os.path.abspath(__file__))

mujoco_temp=args.python_path + " " + _this_file_dir + "/main_ppo.py --algo ppo --use-gae --log-interval 1 " \
                      "--num-steps 2000 --num-processes 1 --lr 5e-5 --entropy-coef 0 " \
                      "--clip-param 0.1 --value-loss-coef 0.5 --ppo-epoch 4 --num-mini-batch 32 " \
                      "--hidden-size 150 " \
                      "--gamma 0.99 --gae-lambda 0.95 --num-env-steps 20000000 --no-cuda --use-proper-time-limits " \
                      "--seed {3} " \
                      "--env-name {0} "

temp=mujoco_temp
config={"session_name":"run-all-"+args.env_name,"windows":[]}

for i in range(args.seeds):
    panel_list=[]
    env_name=args.env_name
    env_name_prefix_dir=env_name.split('-')[0]
    experiment_count=i
    random_seed=i+1000
    panel_list.append(temp.format(env_name,env_name_prefix_dir,experiment_count,random_seed))
    config["windows"].append({"window_name":"seed-{}".format(i),"panels":panel_list})

file_name=args.yaml.split(".")[0]+".yaml"
yaml.dump(config,open(file_name,"w"),default_flow_style=False)

