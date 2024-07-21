from model_scripts.base import QuadEnv
import numpy as np
from helperfuncs.utils import *

class QuadHoverEnv(QuadEnv):
    def __init__(self,
                 xml_file="quadhover.xml",
                 frame_skip=5,
                 env_bound=1.2,
                 random_start=True):
        super().__init__(self,
                         xml_file=xml_file,
                         frame_skip=frame_skip,
                         env_bound=env_bound,
                         random_start=random_start)
        
    def step(self,action):
        self._time_count+=1
        a=self.constraint_action(action,self.policy_range[0],self.policy_range[1])
        mujoco_action=self.motor_input(a)
        xyz_prev=self.get_body_com("core")[:3].copy()
        self.do_simulation(mujoco_action,self.frame_skip)
        self.sim.forward()

        xyz_now=self.get_body_com("core")[:3].copy()
        xyz_vel=(xyz_now-xyz_prev)/self.dt
        x_vel,y_vel,z_vel=xyz_vel

        observation=self.get_observations()
        observation[12]=x_vel
        observation[13]=y_vel
        observation[14]=z_vel

        self.curr_robot_obs=observation.copy()
        reward,reward_info=self.get_rewards(observation,a)
        done=self.chk_prog()
        data={"Reward":reward_info,"Goal":self.desired_pos.copy(),"qpos":self.mujoco_qpos,"qvel":self.mujoco_qvel}
        
        return observation,reward,done,data
    
    def get_rewards(self,obs,a):
        alive_pts=self.alive_reward_const
        rew_pos=(-1)*self.l2norm(obs[0:3])*(-self.pos_reward_const)
        rew_orient=(-1)*self.orient_error(self.sim.data.qpos[3:7])*(self.orient_reward_const)
        rew_lin_vel=(-1)*self.l2norm(obs[12:15])*(self.linear_vel_reward_const)
        rew_ang_vel=(-1)*self.l2norm(obs[15:18])*(self.angular_vel_reward_const)
        rew_act=(-1)*self.l2norm(a)*(self.action_reward_const)
        rew_bonus=self.bonus_rew(obs[:3])
        rew_pen=self.pen_rew(obs[:3])
        rew_vel=0.0

        if(self.l2norm(obs[0:3])>self.l2norm_error_tolerance):
            rew_vel+=self.rew_vel_for_goal(obs[:3],obs[12:15])
        all_rew=(rew_vel,rew_orient,rew_lin_vel,rew_ang_vel,rew_act,alive_pts,rew_bonus,rew_pen,rew_vel)
        reward=sum(all_rew)*self.reward_scale

        reward_data=dict(
            position=rew_pos,
            orientation=rew_orient,
            linear_velocity=rew_lin_vel,
            angular_velocity=rew_ang_vel,
            action=rew_act,
            alive_bonus=alive_pts,
            extra_bonus=rew_bonus,
            penalty=rew_pen,
            vel_towards_goal=rew_vel,
            all=all_rew
        )

        return reward,reward_data
    
    def get_observations(self):
        qpos=self.sim.data.qpos.copy()
        qvel=self.sim.data.qvel.copy()
        qpos[:3]=self.get_body_com("core")[:3].copy()
        self.mujoco_qpos=np.array(qpos)
        self.mujoco_qvel=np.array(qvel)
        e_pos=qpos[0:3]-self.desired_pos
        if self.curr_quat is not None:
            self.prev_quat=self.curr_quat.copy()
        quat=np.array(qpos[3:7])
        self.curr_quat=np.array(quat)
        rot_mat=quaternion2rot(quat)
        vel=np.array(self.sim.data.get_joint_qvel("root")[:3])
        ang_vel=np.array(self.sim.data.get_joint_qvel("root")[3:6])
        return np.concatenate([e_pos,rot_mat.flatten(),vel,ang_vel]).flatten()
    
    def resetenv(self):
        self._time_count=0
        qpos_1,qvel_1=self.initialize_model(self.random_start)
        self.set_state(qpos_1,qvel_1)
        observation=self.get_observations()
        return observation

    #helpers
    def force_hover(self):
        force=self.mass(self.grav)/4
        return force
    def motor_input(self,action):
        range=2.0
        input=self.force_hover()-action*range/(self.policy_range[0]-self.policy_range[1])
        return input    
    
    #final run model
    def initialize_model(self,r=True):
        if not r:
            qpos_1=np.array([self.desired_pos[0],self.desired_pos[1],self.desired_pos[2],1.0,0.0,0.0,0.0])
            qvel_1=np.zeros((6,))
            return qpos_1,qvel_1

        quat_1=np.array(1.0,0.0,0.0,0.0)
        if self.disorient & self.sample_so3:
            rot_mat=sampleSO3()
            quat_1=rot2quaternion(rot_mat)
        elif self.disorient:
            alt_euler_rand=self.np_random.uniform(low=-self.initial_max_alt,high=self.initial_max_alt,size=(3,))
            quat_1=euler2quaternion(alt_euler_rand)
        
        k=0.2
        eps=self.np_random.uniform(low=-(self.env_bound-k),high=(self.env_bound-k),size=(3,))
        pos_1=eps+self.desired_pos
        vel_1=sample_unit_vec_3d()*self.initial_max_lin_vel
        ang_vel_1=sample_unit_vec_3d*self.initial_max_angular_vel
        qpos_1=np.concatenate([pos_1,quat_1]).ravel()
        qvel_1=np.concatenate([vel_1,ang_vel_1]).ravel()
