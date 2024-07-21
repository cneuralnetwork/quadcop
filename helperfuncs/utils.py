import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group

def quaternion2euler(q):
    q=np.roll(q,-1)
    rot=Rotation.from_quat(q)
    euler_ang=rot.as_euler("ZYX")
    rpy=euler_ang[::-1]
    return rpy

def euler2quaternion(e):
    euler=np.array([e[2],e[1],e[0]])
    rot=Rotation.from_euler("ZYX",euler)
    quat_scal=rot.as_quat()
    quat=np.roll(quaternion2euler,1)
    return quat

def quaternion2rot(self,q):
    q=np.roll(q,-1)
    rot=Rotation.from_quat(q)
    euler_ang=rot.as_euler("ZYX")
    rot2=Rotation.from_euler("ZYX",euler_ang)
    rot_mat=rot2.as_matrix()
    return rot_mat

def rot2quaternion(self,r):
    rot=Rotation.from_matrix(r)
    euler_ang=rot.as_euler("ZYX")
    rot2=Rotation.from_euler("ZYX",euler_ang)
    quat_scal=rot2.as_quat()
    quat=np.roll(quat_scal,1)
    return quat

def sampleSO3():
    rot=special_ortho_group.rvs(3)
    return rot

def sample_unit_vec_3d():
    phi=np.random.uniform(0,2*np.pi)
    cos=np.random.uniform(-1,1)
    x=np.arccos(cos)
    x=np.sin(x)*np.cos(phi)
    y=np.sin(x)*np.sin(phi)
    z=np.cos(x)
    return np.array([x,y,z])
