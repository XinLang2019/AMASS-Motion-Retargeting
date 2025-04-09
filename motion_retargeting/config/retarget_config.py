import torch

class G1Cfg:
    G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
   
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])
    
    mjcf_file = "resources/robots/g1/g1_29dof_anneal_23dof.xml"
    
    
class H1Cfg:
    H1_ROTATION_AXIS = torch.tensor([[
    [0, 0, 1], # l_hip_yaw
    [1, 0, 0], # l_hip_roll
    [0, 1, 0], # l_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # r_hip_yaw
    [1, 0, 0], # r_hip_roll
    [0, 1, 0], # r_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # torso
    
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_roll_pitch
    [0, 0, 1], # l_yaw_pitch
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_roll_pitch
    [0, 0, 1], # r_yaw_pitch
    
    [0, 1, 0], # r_elbow
    ]])
    
    mjcf_file = "resources/robots/h1/h1.xml"