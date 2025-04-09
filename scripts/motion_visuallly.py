"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import argparse
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from motion_retargeting.utils.motion_lib_robot import MotionLibRobot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from motion_retargeting.utils.flags import flags


flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments



def main():
    # parse arguments
    cfg = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                                custom_parameters=[{
                                    "name": "--asset_id",
                                    "type": int,
                                    "default": 0,
                                    "help": "Asset id"
                                }, {
                                    "name": "--speed_scale",
                                    "type": float,
                                    "default": 1.0,
                                    "help": "Animation speed scale"
                                }, {
                                    "name": "--show_axis",
                                    "action": "store_true",
                                    "help": "Visualize DOF axis"
                                },{
                                    "name": "--robot_type",
                                    "type": str,
                                    "default": "g1",  # 可以根据需要设置默认值
                                    "help": "Specify the robot type (e.g., g1 or h1)"
                                }])
    robot_type = cfg.robot_type
    if robot_type == "g1":
        g1_xml = "resources/robots/g1/g1_29dof_anneal_23dof.xml"
        g1_urdf = "resources/robots/g1/g1_29dof_anneal_23dof.urdf"
        motion_file = "data/g1/test.pkl"
    elif robot_type == 'h1':
        g1_xml = "resources/robots/h1/h1.xml"
        g1_urdf = "resources/robots/h1/urdf/h1.urdf"
        motion_file = "data/h1/test.pkl"

    asset_descriptors = [
        AssetDesc(g1_urdf, False),
    ]
    sk_tree = SkeletonTree.from_mjcf(g1_xml)

    if os.path.exists(motion_file):
        print(f"loading {motion_file}")
    else:
        raise ValueError(f"Motion file {motion_file} does not exist! Please run grad_fit_g1.py first.")
    
    if cfg.asset_id < 0 or cfg.asset_id >= len(asset_descriptors):
        print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
        quit()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if cfg.physics_engine == gymapi.SIM_FLEX:
        pass
    elif cfg.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = cfg.num_threads
        sim_params.physx.use_gpu = cfg.use_gpu
        sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline

    if not cfg.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(cfg.compute_device_id, cfg.graphics_device_id, cfg.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load asset
    asset_root = "./"
    asset_file = asset_descriptors[cfg.asset_id].file_name

    asset_options = gymapi.AssetOptions()
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set up the env grid
    num_envs = 1
    num_per_row = 5
    spacing = 5
    env_lower = gymapi.Vec3(-spacing, spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    num_dofs = gym.get_asset_dof_count(asset)
    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # set default DOF positions
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    gym.prepare_sim(sim)

    device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

    motion_lib = MotionLibRobot(robot_type=robot_type, motion_file=motion_file, device=device, masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=g1_xml)
    num_motions = 1
    curr_start = 0
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
    motion_keys = motion_lib.curr_motion_keys

    current_dof = 0
    speeds = np.zeros(num_dofs)

    time_step = 0
    rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
    rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
    rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    actor_root_state = gymtorch.wrap_tensor(actor_root_state)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "previous")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "add")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "next_batch")
    motion_id = 0
    motion_acc = set()

    env_ids = torch.arange(num_envs).int().to(cfg.sim_device)

    ## Create sphere actors
    radius = 0.1
    color = gymapi.Vec3(1.0, 0.0, 0.0)
    sphere_params = gymapi.AssetOptions()

    sphere_asset = gym.create_sphere(sim, radius, sphere_params)

    num_spheres = 19
    init_positions = gymapi.Vec3(0.0, 0.0, 0.0)
    spacing = 0.

    while not gym.query_viewer_has_closed(viewer):
        # step the physics

        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
    
        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(cfg.compute_device_id), torch.tensor([motion_time]).to(cfg.compute_device_id))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        if cfg.show_axis:
            gym.clear_lines(viewer)
            
        gym.clear_lines(viewer)
        gym.refresh_rigid_body_state_tensor(sim)
        # import pdb; pdb.set_trace()
        idx = 0
        for pos_joint in rb_pos[0, 1:]: # idx 0 torso (duplicate with 11)
            sphere_geom2 = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0.0, 0.0))
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
            gymutil.draw_lines(sphere_geom2, gym, viewer, envs[0], sphere_pose) 
    
        ###########################################################################
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
        # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.refresh_actor_root_state_tensor(sim)

        # dof_pos = dof_pos.cpu().numpy()
        # dof_states['pos'] = dof_pos
        # speed = speeds[current_dof]
        dof_state = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1).squeeze().repeat(num_envs, 1)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)
        
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        # time_step += 1/5
        time_step += dt

        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "previous" and evt.value > 0:
                motion_id = (motion_id - 1) % num_motions
                print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
            elif evt.action == "next" and evt.value > 0:
                motion_id = (motion_id + 1) % num_motions
                print(f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}")
            elif evt.action == "add" and evt.value > 0:
                motion_acc.add(motion_keys[motion_id])
                print(f"Adding motion {motion_keys[motion_id]}")
            elif evt.action == "print" and evt.value > 0:
                print(motion_acc)
            elif evt.action == "next_batch" and evt.value > 0:
                curr_start += num_motions
                motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
                motion_keys = motion_lib.curr_motion_keys
                print(f"Next batch {curr_start}")

            time_step = 0
    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    # args = parser.parse_args()
    
    main()