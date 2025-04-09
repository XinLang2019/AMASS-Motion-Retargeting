import glob
import os
import sys
import pdb
import argparse
import os.path as osp
sys.path.append(os.getcwd())

from motion_retargeting.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from motion_retargeting.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from motion_retargeting.utils.torch_robot_humanoid_batch import Humanoid_Batch
from motion_retargeting.config.retarget_config import G1Cfg, H1Cfg

def robot_shape(args):
    robot_type = args.robot_type
    
    if robot_type == 'g1':
        rotation_axis = G1Cfg.G1_ROTATION_AXIS
        mjcf_file = G1Cfg.mjcf_file
    elif robot_type == 'h1':
        rotation_axis = H1Cfg.H1_ROTATION_AXIS
        mjcf_file = H1Cfg.mjcf_file
    
    robot_fk = Humanoid_Batch(args.robot_type, mjcf_file, extend_head=True) # load forward kinematics model
    robot_joint_names = robot_fk.model_names
    
    #### Define corresonpdances between robot and smpl joints
    robot_joint_names_augment = robot_joint_names + ["head_link"]
    if robot_type == 'g1':
        robot_joint_pick = ['pelvis',  'left_hip_pitch_link', "left_knee_link", "left_ankle_pitch_link",  'right_hip_pitch_link', 'right_knee_link', 'right_ankle_pitch_link', "left_shoulder_pitch_link", "left_elbow_link", "left_hand_link", "right_shoulder_pitch_link", "right_elbow_link", "right_hand_link", "head_link"]
    elif robot_type == 'h1':
        robot_joint_pick = ['pelvis',  'left_hip_yaw_link', "left_knee_link", "left_ankle_link",  'right_hip_yaw_link', 'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link", "head_link"]

    smpl_joint_pick = ["Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Head"]
    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    # print("############", len(robot_joint_names_augment))
    #### Preparing fitting varialbes
    device = torch.device("cpu")
    if robot_type == 'g1':
        pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 27, axis = 2), 1, axis = 1)
        dof_pos = torch.zeros((1, 23))
    elif robot_type == 'h1':
        pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), 1, axis = 1)
        dof_pos = torch.zeros((1, 19))
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    pose_aa_robot = torch.cat([torch.zeros((1, 1, 3)), rotation_axis * dof_pos[..., None], torch.zeros((1, 2, 3))], axis = 1)
    root_trans = torch.zeros((1, 1, 3))    

    ###### prepare SMPL default pause for robot
    pose_aa_stand = np.zeros((1, 72))
    rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
    pose_aa_stand[:, :3] = rotvec
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = robot_fk.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])

    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)

    for iteration in range(1000):
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        loss_g = diff.norm(dim = -1).mean() 
        loss = loss_g
        if iteration % 100 == 0:
            print(iteration, loss.item() * 1000)

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    os.makedirs("data/" + robot_type, exist_ok=True)
    joblib.dump((shape_new.detach(), scale), "data/" + robot_type + "/shape_optimized_v1.pkl") # V2 has hip jointsrea
    print(f"shape fitted and saved to data/{robot_type}/shape_optimized_v1.pkl")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()
    
    robot_shape(args)