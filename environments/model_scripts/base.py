import os
from abc import ABC
from typing import Tuple,Dict
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env

class QuadEnv(mujoco_env.MujocoEnv,utils.EzPickle,ABC):
    action_thrust=np.arange(0,4)
    obs_xyz=np.arange(0,3)
    obs_rot_mat=np.arange(3,12)
    obs_vel=np.arange(12,15)
    obs_vel_a=np.arange(15,18)

    def __init__(self,
                 xml_file="quadhover.xml",
                 frame_skip=5,
                 error_tolerance=0.05,
                 max_time_steps=1000,
                 random_start=True,
                 disorient=True,
                 obs_noise=0,
                 env_bound=1.2,
                 initial_max_vel=0.5,
                 initial_max_angular_vel=0.1*np.pi,
                 initial_max_alt=np.pi/3.0,
                 bonus_reach=15.0,
                 max_reward=2.0,
                 pos_reward_const=5.0,
                 orient_reward_const=0.02,
                 linear_vel_reward_const=0.01,
                 angular_vel_reward_const=0.001,
                 action_reward_const=0.0025,
                 alive_reward_const=5.0,
                 reward_scale=1.0):
        project_name=""
        xml_path=project_name+"environments/model_scripts/base.py"
        self.reward_scale=reward_scale
        self.alive_reward_const=alive_reward_const
        self.action_reward_const=action_reward_const
        self.angular_vel_reward_const=angular_vel_reward_const
        self.linear_vel_reward_const=linear_vel_reward_const
        self.orient_reward_const=orient_reward_const
        self.pos_reward_const=pos_reward_const
        self.max_reward=max_reward
        self.bonus_reach=bonus_reach
        self.initial_max_alt=initial_max_alt
        self.initial_max_angular_vel=initial_max_angular_vel
        self.initial_max_vel=initial_max_vel
        self.env_bound=env_bound
        self.obs_noise=obs_noise
        self.disorient=disorient
        self.random_start=random_start
        self.max_time_steps=max_time_steps
        self.error_tolerance=error_tolerance
        self.frame_skip=frame_skip

        #additional stuff
        self.policy_range=[-1.0,1.0]
        self.safe_policy_range=[-0.8,0.8]
        self.curr_policy_at=np.array([-1,-1,-1,-1.])

        #positions and velocities
        self.desired_pos=np.array([0,0,3.0])
        self.mujoco_qpos=None
        self.mujoco_qvel=None
        self.prev_robot_obs=None
        self.curr_robot_obs=None

        self._time_count=0.0
        self.grav=-9.81

        #final stuff
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip)
        self.grav=float(abs(self.model.opt.gravity[2]))

    #helpers
    def l2norm(self,x:np.ndarray)->float:
        return np.linalg.norm(x)
    def get_mujoco_obs(self)->np.ndarray:
        qpos=self.sim.data.qpos.copy()
        qvel=self.sim.data.qvel.copy()
        self.mujoco_qpos=np.array(qpos)
        self.mujoco_qvel=np.array(qvel)
        final_obs=np.concatenate([qpos,qvel]).flatten()
        return final_obs
    def mass(self)->float:
        return self.model.body_mass[1] 
    def inertia(self)->np.ndarray:
        return np.array(self.sim.data.cinert)
    def joints_qpos_qvel(self,joint:str)->Tuple[np.ndarray,np.ndarray]:
        qpos=self.data.get_joint_qpos(joint).copy()
        qvel=self.data.get_joint_qvel(joint).copy()
        return qpos,qvel
    def get_data(self):
        print("Mass : ",self.mass)
        print("Minimum Action : ",self.action_space.low)
        print("Maximum Action : ",self.action_space.high)
        print("Actuator Control : ",type(self.model.actuator_ctrlrange))
        print("Actuator Force Range : ",self.model.actuator_forcerange)
        print("Actuator Force Limited : ",self.model.actuator_forcelimited)
        print("Actuator Control Limited : ",self.model.actuator_ctrllimited)
        
    
    #norm of bounds and error
    def l2norm_bounding_box(self)->float:
        k=np.array([self.env_bound,self.env_bound,self.env_bound])
        return self.l2norm(k)
    def l2norm_error_tolerance(self)->float:
        k=np.array([self.error_tolerance,self.error_tolerance,self.error_tolerance])
        return self.l2norm(k)
    
    #step in env
    def step(self,x:np.ndarray):
        rew=0
        observations=self.get_mujoco_obs()
        check_not_done=np.isfinite().all()
        check_done= not check_not_done
        data={"Reward":rew,"qpos":self.mujoco_qpos,"qvel":self.mujoco_qvel}
        return observations,rew,check_done,data
    
    #use relevant action data
    def constraint_action(self,action:np.ndarray,min=-1.0,max=1.0)->np.ndarray:
        return np.clip(action,a_min=min,a_max=max)
    
    #check goal-curr ~ 0
    def goal_check(self,err:np.ndarray)->bool:
        return self.l2norm(err)<self.l2norm_error_tolerance()
    
    #check if drone is in env bound
    def check_bound(self,err:np.ndarray)->bool:
        return self.l2norm(err)<self.l2norm_bounding_box
    
    







