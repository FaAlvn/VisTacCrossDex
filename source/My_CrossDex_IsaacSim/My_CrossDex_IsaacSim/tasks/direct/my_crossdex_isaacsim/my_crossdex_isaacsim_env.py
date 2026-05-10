# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply
from isaacsim.core.cloner import Cloner

from isaaclab.sim.utils import create_prim
from .my_crossdex_isaacsim_env_cfg import MyCrossdexIsaacsimEnvCfg

import sys
import os
sys.path.append('../..')
# Get the path to the scripts directory
retargeting_dir = os.path.join("/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/retargeting/")
if retargeting_dir not in sys.path:
    sys.path.append(retargeting_dir)

from retargeting.retargeting_nn_utils import EigenRetargetModel

class MyCrossdexIsaacsimEnv(DirectRLEnv):
    cfg: MyCrossdexIsaacsimEnvCfg

    def __init__(self, cfg: MyCrossdexIsaacsimEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.device = "cuda"

        # self.num_envs = self.cfg.num_envs
        self.num_agents = 1

        self.num_states = self.cfg.num_states
        self.num_observations = self.cfg.num_observations
        
        self.observation_space = self.cfg.observation_space
        self.state_space = self.cfg.state_space

        self.reward_function = self.cfg.reward_function

        self.num_actions = self.cfg.num_actions
        self.control_freq_inv = self.cfg.control_freq_inv

        self.action_space = self.cfg.action_space
        
        self.clipobs = self.cfg.clipObservations
        self.clip_actions = self.cfg.clipActions
        self.success_tolerance = self.cfg.success_tolerance

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps = self.cfg.render_fps
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        
        self.allocate_buffers()
        self.obs_dict = {}

        self.av_factor = self.cfg.av_factor
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cont_success_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        self.reward_function = self.cfg.reward_function
        # self.max_episode_length = self.cfg.max_episode_length
        self.reset_time = self.cfg.reset_time
        self.maxConsecutiveSuccesses = self.cfg.maxConsecutiveSuccesses
        self.goal_height = self.cfg.goal_height
        self.multi_task_label = self.cfg.multi_task_label

        self.hand_names = self.cfg.hand_names
        self.num_robots = self.cfg.num_robots
        self.arm_dof_names = self.cfg.arm_dof_names
        self.hand_dof_names = self.cfg.hand_dof_names
        self.palm_offset = self.cfg.palm_offset
        self.num_hand_dofs = self.cfg.num_hand_dofs
        self.num_arm_dofs = self.cfg.num_arm_dofs
        self.num_fingers = self.cfg.num_fingers
        self.agg_num_fingers = self.cfg.agg_num_fingers
        self.agg_num_envs = self.num_envs // len(self.hand_names)
        # self.num_hand_dofs = self.cfg.num_hand_dofs
        # self.num_arm_dofs = self.cfg.num_arm_dofs
        self.dataset = self.cfg.dataset
        self.add_random_dataset = self.cfg.add_random_dataset
        self.num_eigengrasp_actions = self.cfg.num_eigengrasp_actions
        self.arm_dof = self.cfg.arm_dof
        self.num_rl_actions = self.cfg.num_rl_actions

        # self.object_start_pos = self.cfg.object_start_pos
        self.object_start_poses = self.cfg.object_start_poses
        # self.object_init_state = self.cfg.object_init_state
        self.object_init_states = self.cfg.object_init_states

        self.actions_eigengrasp = torch.zeros(self.num_envs,self.num_eigengrasp_actions).to(self.device)
        self.init_retargeting()

        self.test_forced_lift = self.cfg.test_forced_lift
        self.test = self.cfg.test
        self.n_lift_steps = self.cfg.n_lift_steps
        
        self.lift_step_count = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.is_lifting_stage = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        
        
        self.object_assets_cfg = self.cfg.object_assets_cfg
        

        if self.test:
            self.lift_arm_dof_per_robot = [torch.zeros((self.agg_num_envs,n),dtype=torch.float32).to(self.device) \
                for n in self.num_arm_dofs] #[self.lift_arm_target.unsqueeze(0).repeat(self.agg_num_envs,1) for n in self.num_arm_dofs]
            self.lift_hand_dof_per_robot = [torch.zeros((self.agg_num_envs,n),dtype=torch.float32).to(self.device) \
                for n in self.num_hand_dofs]
            #for a,b in zip(self.lift_arm_dof_per_robot, self.lift_hand_dof_per_robot):
            #    print(a.shape, b.shape)
            self.lift_step_count = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            self.is_lifting_stage = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            self.reward_additional_params = dict(
                test_forced_lift = self.test_forced_lift,
                n_lift_steps = self.n_lift_steps,
                lift_step_count = self.lift_step_count,
                is_lifting_stage = self.is_lifting_stage,
            )
        else:
            self.reward_additional_params = dict(test_forced_lift = self.test_forced_lift)

    def init_retargeting(self):
        retargeting_type = "dexpilot"
        self.retargeting_models, self.retarget2isaacs = [], []
        for i_robot, hand_name in enumerate(self.hand_names):
            retargeting_model = EigenRetargetModel(
                self.dataset, 
                hand_name, 
                self.add_random_dataset, 
                retargeting_type,
                device=self.device,
                n_eigengrasps=self.num_eigengrasp_actions
            )
            self.retargeting_models.append(retargeting_model)
            print("names:", retargeting_model.robot_joint_names)
            retarget2isaac = [retargeting_model.robot_joint_names.index(n) for n in self.hand_dof_names[i_robot]]
            self.retarget2isaacs.append(retarget2isaac)
            print("\nRobot:", hand_name)
            print("Retarget2isaac dof idx mapping:", retarget2isaac)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        pass
        # allocate buffers
        # self.obs_buf = torch.zeros(
        #     (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # self.states_buf = torch.zeros(
        #     (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        # self.rew_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.float)
        # self.reset_buf = torch.ones(
        #     self.num_envs, device=self.device, dtype=torch.long)
        # self.timeout_buf = torch.zeros(
        #      self.num_envs, device=self.device, dtype=torch.long)
        # self.progress_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.long)
        # self.randomize_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.long)
        # self.extras = {}
        # if hasattr(self, "num_student_observations"):
        #     self.student_obs_buf = torch.zeros((self.num_envs, self.num_student_observations), device=self.device, dtype=torch.float)

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.device)

        return actions
    
    def _setup_scene(self):
        self.robots = [Articulation(self.cfg.robot_assets_cfg[i]) for i in range(self.cfg.num_robots)]
        self.objects = [RigidObject(self.cfg.object_assets_cfg[i]) for i in range(self.cfg.num_robots)]
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        for robot_idx in range(self.cfg.num_robots):
            self.scene.articulations[self.cfg.hand_names[robot_idx]] = self.robots[robot_idx]
            self.scene.rigid_objects[self.cfg.hand_names[robot_idx]] = self.objects[robot_idx]

        
        cloner = Cloner()
        
        leap_hand_target_paths = cloner.generate_paths("/World/envs/leap_hand/env", int(self.cfg.num_envs/self.cfg.num_robots))
        shadow_hand_target_paths = cloner.generate_paths("/World/envs/shadow_hand/env", int(self.cfg.num_envs/self.cfg.num_robots))
        inspire_hand_target_paths = cloner.generate_paths("/World/envs/inspire_hand/env", int(self.cfg.num_envs/self.cfg.num_robots))
        schunk_svh_hand_target_paths = cloner.generate_paths("/World/envs/schunk_svh_hand/env", int(self.cfg.num_envs/self.cfg.num_robots))
        
        self.leap_positions = np.array([[i*2, 0, 0] for i in range(-int(self.cfg.num_envs/self.cfg.num_robots/2), int(self.cfg.num_envs/self.cfg.num_robots/2))])
        self.shadow_positions = np.array([[i*2, 2, 0] for i in range(-int(self.cfg.num_envs/self.cfg.num_robots/2), int(self.cfg.num_envs/self.cfg.num_robots/2))])
        self.inspire_positions = np.array([[i*2, 4, 0] for i in range(-int(self.cfg.num_envs/self.cfg.num_robots/2), int(self.cfg.num_envs/self.cfg.num_robots/2))])
        self.schunk_svh_positions = np.array([[i*2, 6, 0] for i in range(-int(self.cfg.num_envs/self.cfg.num_robots/2), int(self.cfg.num_envs/self.cfg.num_robots/2))])        
        self.env_origins = [self.leap_positions, self.shadow_positions, self.inspire_positions, self.schunk_svh_positions]
        cloner.clone(source_prim_path="/World/envs/leap_hand/env_0", prim_paths=leap_hand_target_paths, positions = self.leap_positions)
        cloner.clone(source_prim_path="/World/envs/shadow_hand/env_0", prim_paths=shadow_hand_target_paths, positions = self.shadow_positions)
        cloner.clone(source_prim_path="/World/envs/inspire_hand/env_0", prim_paths=inspire_hand_target_paths, positions = self.inspire_positions)
        cloner.clone(source_prim_path="/World/envs/schunk_svh_hand/env_0", prim_paths=schunk_svh_hand_target_paths, positions = self.schunk_svh_positions)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: arm_dof + eigengrasp_actions
        actions_eigengrasp = actions[:,self.arm_dof:]
        actions_arm = actions[:,0:self.arm_dof]
        # print("ACTIONS:", actions.shape)
        actions_dof = torch.zeros((self.cfg.num_envs, self.arm_dof + max(self.num_hand_dofs))).to(self.device)
        # print("actions_dof:", actions_dof.shape)
        for robot_idx, robot in enumerate(self.robots):
            robot_actions_eigengrasp = actions_eigengrasp[int(self.cfg.num_envs/self.num_robots)*robot_idx:int(self.cfg.num_envs/self.num_robots)*(robot_idx+1), :]
            robot_actions_hand_dof = self.retargeting_models[robot_idx].retarget(robot_actions_eigengrasp)[:, self.retarget2isaacs[robot_idx]]
            robot_actions_arm = actions_arm[int(self.cfg.num_envs/self.num_robots)*robot_idx:int(self.cfg.num_envs/self.num_robots)*(robot_idx+1), :]

            robot_actions_dof = torch.zeros(int(self.num_envs/self.num_robots),self.arm_dof+self.num_hand_dofs[robot_idx]).to(self.device)
            robot_actions_dof[:,0:self.arm_dof] = robot_actions_arm
            robot_actions_dof[:,self.arm_dof:] = robot_actions_hand_dof
            # print("robot_actions_dof:", robot_actions_dof.shape)
            actions_dof[int(self.cfg.num_envs/self.num_robots)*robot_idx:int(self.cfg.num_envs/self.num_robots)*(robot_idx+1), 0:self.arm_dof+self.num_hand_dofs[robot_idx]] = robot_actions_dof

        # print("actions_dof:", actions_dof.shape)
        self.actions = actions_dof.clone() # arm_dof + hand_dof
        # print("actions:", self.actions)
        self.actions_eigengrasp = actions_eigengrasp
         
    def _apply_action(self) -> None:
        # print("self.actions:", np.shape(self.actions))
        for robot_idx, robot in enumerate(self.robots):
            robot_actions = self.actions[int(self.cfg.num_envs/self.num_robots)*robot_idx:int(self.cfg.num_envs/self.num_robots)*(robot_idx+1), 0:self.arm_dof+self.num_hand_dofs[robot_idx]]
            # print("robot_actions:", robot_actions.shape)
            robot.set_joint_position_target(robot_actions)
            robot.set_joint_position_target(robot_actions)#self.actions
            robot.write_data_to_sim()
        
    def get_state(self):
        # print("SHAPE:", np.shape(self.robots[0].data.joint_pos))
        # print("SHAPE:", np.shape(self.robots[1].data.joint_pos))
        # print("SHAPE:", np.shape(self.robots[2].data.joint_pos))
        # print("SHAPE:", np.shape(self.robots[3].data.joint_pos))

        q = torch.empty((0, self.arm_dof)).to(self.device)
        palm_pos = torch.empty((0, 3)).to(self.device)
        thumb_pos = torch.empty((0, 3)).to(self.device)
        index_finger_pos = torch.empty((0, 3)).to(self.device)
        middle_finger_pos = torch.empty((0, 3)).to(self.device)
        ring_finger_pos = torch.empty((0, 3)).to(self.device)
        palm_and_finger_pos = torch.empty((0, 15)).to(self.device)
        fingertip_pos = torch.empty((int(self.num_envs/self.num_robots), 0, 3)).to(self.device)
        palm_rot = torch.empty((0, 4)).to(self.device)
        palm_center_pos = torch.empty((0, 3)).to(self.device)
        obj_pose = torch.empty((0,7)).to(self.device)
        object_linvel = torch.empty((0,3)).to(self.device)
        object_angvel = torch.empty((0,3)).to(self.device)
        for robot_idx, robot in enumerate(self.robots):
            robot_q = robot.data.joint_pos[:,0:self.arm_dof] # arm_dof
            q = torch.cat([q, robot_q], dim=0)
            # print("q:", np.size(q))
            if self.hand_names[robot_idx] == "leap_hand":
                # print("leap")
                palm_idx = robot.find_bodies("base_link")[0]
                thumb_idx = robot.find_bodies("thumb_fingertip")[0]
                index_finger_idx = robot.find_bodies("fingertip")[0]
                middle_finger_idx = robot.find_bodies("fingertip_2")[0]
                ring_finger_idx = robot.find_bodies("fingertip_3")[0]
            elif self.hand_names[robot_idx] == "shadow_hand":
                # print("shadow")
                palm_idx = robot.find_bodies("base_link")[0]
                thumb_idx = robot.find_bodies("thdistal")[0]
                index_finger_idx = robot.find_bodies("ffdistal")[0]
                middle_finger_idx = robot.find_bodies("mfdistal")[0]
                ring_finger_idx = robot.find_bodies("rfdistal")[0]
                pinky_finger_idx = robot.find_bodies("lfdistal")[0]
            elif self.hand_names[robot_idx] == "inspire_hand":
                # print("inspire")
                palm_idx = robot.find_bodies("arm_link6")[0]
                thumb_idx = robot.find_bodies("thumb_distal")[0]
                index_finger_idx = robot.find_bodies("index_intermediate")[0]
                middle_finger_idx = robot.find_bodies("middle_intermediate")[0]
                ring_finger_idx = robot.find_bodies("ring_intermediate")[0]
                pinky_finger_idx = robot.find_bodies("pinky_intermediate")[0]
            elif self.hand_names[robot_idx] == "schunk_svh_hand":
                # print("svh")
                palm_idx = robot.find_bodies("arm_link6")[0]
                thumb_idx = robot.find_bodies("right_hand_c")[0]
                index_finger_idx = robot.find_bodies("right_hand_t")[0]
                middle_finger_idx = robot.find_bodies("right_hand_s")[0]
                ring_finger_idx = robot.find_bodies("right_hand_r")[0]
                pinky_finger_idx = robot.find_bodies("right_hand_l")[0]
            else:
                pass
            # print("body_pos_w:")
            robot_palm_pos = robot.data.body_pos_w[:,palm_idx,:].reshape(-1,3)
            robot_thumb_pos = robot.data.body_pos_w[:,thumb_idx,:].reshape(-1,3)
            robot_index_finger_pos = robot.data.body_pos_w[:,index_finger_idx,:].reshape(-1,3)
            robot_middle_finger_pos = robot.data.body_pos_w[:,middle_finger_idx,:].reshape(-1,3)
            robot_ring_finger_pos = robot.data.body_pos_w[:,ring_finger_idx,:].reshape(-1,3)
            if self.cfg.num_fingers[robot_idx]==5:
                robot_pinky_finger_pos = robot_pinky_finger_pos = robot.data.body_pos_w[:,pinky_finger_idx,:].reshape(-1,3)
            
            robot_palm_and_finger_pos = torch.cat([robot_palm_pos, robot_thumb_pos, robot_index_finger_pos, robot_middle_finger_pos, robot_ring_finger_pos], dim=-1)
            
            # print("robot_palm_pos:", robot_palm_pos.shape)
            # print("robot_thumb_pos:", robot_thumb_pos.shape)
            # print("robot_middle_finger_pos:", robot_middle_finger_pos.shape)
            # print("robot_ring_finger_pos:", robot_ring_finger_pos.shape)
            # print("robot_palm_and_finger_pos:", robot_palm_and_finger_pos.shape)
        
            palm_pos = torch.cat([palm_pos, robot_palm_pos], dim=0)
            thumb_pos = torch.cat([thumb_pos, robot_thumb_pos], dim=0)
            index_finger_pos = torch.cat([index_finger_pos, robot_index_finger_pos], dim=0)
            middle_finger_pos = torch.cat([middle_finger_pos, robot_middle_finger_pos], dim=0)
            ring_finger_pos = torch.cat([ring_finger_pos, robot_ring_finger_pos], dim=0)
            palm_and_finger_pos = torch.cat([palm_and_finger_pos, robot_palm_and_finger_pos], dim=0)
            # print("palm_pos:", palm_pos.shape)
            # print("thumb_pos:", thumb_pos.shape)
            # print("middle_finger_pos:", middle_finger_pos.shape)
            # print("ring_finger_pos:", ring_finger_pos.shape)
            # print("palm_and_finger_pos:", palm_and_finger_pos.shape)
            # print("q:", q.shape)

            robot_palm_rot = robot.data.body_quat_w[:,palm_idx,:].reshape(-1,4)
            robot_palm_center_pos = robot_palm_pos + quat_apply(robot_palm_rot, torch.tensor(self.palm_offset).to(self.device).repeat(int(self.num_envs/self.num_robots),1).reshape(int(self.num_envs/self.num_robots),-1))
            if self.cfg.num_fingers[robot_idx]==4:
                robot_fingertip_pos = torch.stack([robot_thumb_pos,robot_index_finger_pos, robot_middle_finger_pos, robot_ring_finger_pos], dim=1)
            else:
                robot_fingertip_pos = torch.stack([robot_thumb_pos,robot_index_finger_pos, robot_middle_finger_pos, robot_ring_finger_pos, robot_pinky_finger_pos], dim=1)
            # print("ROBOT_FINGERTIP_POS:", robot_fingertip_pos.shape)
            palm_rot = torch.cat([palm_rot, robot_palm_rot], dim=0)
            palm_center_pos = torch.cat([palm_center_pos, robot_palm_center_pos], dim=0)
            fingertip_pos = torch.cat([fingertip_pos, robot_fingertip_pos], dim=1)
            # print("FINGERTIP_POS:", fingertip_pos.shape)

            robot_obj_pose = self.objects[robot_idx].data.root_state_w[:,:7]
            # print("robot_object_pos:", robot_obj_pose.shape)
            obj_pose = torch.cat([obj_pose, robot_obj_pose], dim=0)
            robot_object_linvel = self.objects[robot_idx].data.root_state_w[:,7:10]
            robot_object_angvel = self.objects[robot_idx].data.root_state_w[:,10:13]
            object_linvel = torch.cat([object_linvel, robot_object_linvel], dim=0)
            object_angvel = torch.cat([object_angvel, robot_object_angvel], dim=0)

        prev_actions = self.actions_eigengrasp
        # print("prev_Actios:", prev_actions.shape)
        states = torch.cat([q, palm_and_finger_pos, obj_pose, prev_actions], dim=-1)
        self.object_pose = obj_pose
        self.object_pos = obj_pose[:,:3]
        self.object_rot = obj_pose[:,3:7]
        self.object_linvel = object_linvel
        self.object_angvel = object_angvel
        self.palm_pos = palm_pos
        self.palm_rot = palm_rot
        self.palm_center_pos = palm_center_pos

        self.fingertip_pos = fingertip_pos

        return states 
    
    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )

        obs = self.get_state()
        # print(("observations:", obs))
        observations = {"obs": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )
        
        rewards = self.compute_reward(self.actions)

        return rewards

    def compute_reward(self, actions):
        # print("fingertip_pos in compute_reward:", np.shape(self.fingertip_pos))
        if self.reward_function == "v2":
            # print("fingertip_pos in compute_reward v2:", np.shape(self.fingertip_pos))
            (
                rewards,
                self.reset_buf[:],
                self.episode_length_buf[:], #progress_buf in crossdex repo
                self.successes[:],
                self.current_successes[:],
                self.consecutive_successes[:],
                self.cont_success_steps[:],
                reward_info,
            ) = self.reward_v2(
                reset_buf = self.reset_buf,
                progress_buf = self.episode_length_buf,
                successes = self.successes,
                current_successes = self.current_successes,
                consecutive_successes = self.consecutive_successes,
                max_episode_length = self.max_episode_length,
                object_pos = self.object_pos,
                goal_height = self.goal_height,
                palm_pos = self.palm_center_pos,
                fingertip_pos = self.fingertip_pos,
                num_fingers = self.num_fingers,
                agg_num_fingers = self.agg_num_fingers,
                num_envs = self.num_envs,
                agg_num_envs = self.agg_num_envs,
                num_robots = self.num_robots,
                actions = self.actions,
                #self.dist_reward_scale,
                object_init_states = self.object_init_states,
                #self.action_penalty_scale,
                success_tolerance = self.success_tolerance,
                av_factor = self.av_factor,
                cont_success_steps = self.cont_success_steps,
                ### 9-8: forced lift test
                **self.reward_additional_params
            )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes
        self.extras["cont_success_steps"] = self.cont_success_steps

        # if self.print_success_stat:
        #     self.total_resets = self.total_resets + self.reset_buf.sum()
        #     direct_average_successes = self.total_successes + self.successes.sum()
        #     self.total_successes = (
        #         self.total_successes + (self.successes * self.reset_buf).sum()
        #     )
        #     # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
        #     print(
        #         "Direct average consecutive successes = {:.1f}".format(
        #             direct_average_successes / (self.total_resets + self.num_envs)
        #         )
        #     )
        #     if self.total_resets > 0:
        #         print(
        #             "Post-Reset average consecutive successes = {:.1f}".format(
        #                 self.total_successes / self.total_resets
        #             )
        #         )

        return rewards
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel
        # obj_height = self.object.data.root_pos_w[:,2]
        # done = obj_height>0.2
        # done = obj_height<0
        # print("DONE:", done)
        dones = self.successes
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # print("dones:", dones)
        # print("time_out:", time_out)
        # time_out = self.episode_length_buf < self.max_episode_length - 1
        # print("TIME_OUT:", time_out)
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        # return out_of_bounds, time_out
        return dones, time_out

    # def reset(self):
    #     # print("RESET")
    #     self._reset_idx(None)
    #     return self._get_observations()
########################################################################################################################## 
    # def _reset_idx(self, env_ids: Sequence[int] | None):
        
    #     if env_ids is None:
    #         env_ids = self.robot._ALL_INDICES
        
    #     root_states = torch.tensor(self.object_init_states[5], device=self.device).repeat(len(env_ids),1)
    #     root_states[:, :3] += self.scene.env_origins[env_ids]
    #     print("env_origins:", self.scene.env_origins[env_ids])
        
    #     for robot_idx, robot in enumerate(self.robots):
    #         root_states = torch.tensor(self.object_init_states[5], device=self.device).repeat(len(env_ids),1)
    #         root_states[:, :3] += self.env_origins[robot]
    #         self.objects[robot_idx].write_root_pose_to_sim(
    #             root_states, env_ids = env_ids)
        
    #         # reset_robot
    #         robot.write_joint_state_to_sim(
    #             robot.data.default_joint_pos[env_ids],
    #             robot.data.default_joint_vel[env_ids],
    #             env_ids=env_ids
    #         )

##########################################################################################################################


        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    # def get_state(self):

    def reset_done(self):
        # "RESET_DONE"
        done_env_ids = self._get_dones()
        if len(done_env_ids) > 0:
            done_env_ids, time_out = self._reset_idx(done_env_ids)
            
        return self.get_state, done_env_ids
        
    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        self.total_train_env_frames = env_frames
        # print(f'env_frames updated to {self.total_train_env_frames}')

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None
    
    def set_env_state(self, env_state):
        pass

    
    

    def reward_v2(
        self,
        reset_buf,
        progress_buf,
        successes,
        current_successes,
        consecutive_successes,
        max_episode_length: float,
        object_pos,
        goal_height: float,
        palm_pos,
        fingertip_pos,
        num_fingers,
        agg_num_fingers,
        num_envs, 
        agg_num_envs,
        num_robots,
        actions,
        object_init_states,
        success_tolerance: float,
        av_factor: float,
        cont_success_steps,
        K_cont_success_steps = 60,
        max_horizontal_offset = 0.4,
        **kwargs,
    ):
        info = {}

        ### 9-8: forced lift test
        test_forced_lift = kwargs["test_forced_lift"]
        if test_forced_lift:
            n_lift_steps = kwargs["n_lift_steps"]
            lift_step_count = kwargs["lift_step_count"]
            is_lifting_stage = kwargs["is_lifting_stage"]
        
        goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
        palm_object_dist = torch.norm(object_pos - palm_pos.reshape(num_envs,-1), dim=-1)
        palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
        horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1) # horizontally close to (0,0)

        fingertips_object_dist = [] #torch.zeros_like(goal_object_dist)
        for i in range(num_robots):
            offset = agg_num_fingers[i]
            n_f = num_fingers[i]
            # print("offset:", offset)
            # print("fingertip_pos:", np.shape(fingertip_pos[:,offset:offset+n_f]))
            # print("object_pos:", np.shape(object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1)))
            # print("agg_num_fingers:", agg_num_fingers)
            # print("num_fingers:", num_fingers)
            # print("A:", fingertip_pos[:,offset:offset+n_f].shape)
            # print("B:", object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1).shape)
            dists = torch.norm(fingertip_pos[:,offset:offset+n_f] - object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1), dim=-1) # [agg_env,n_finger]
            # print("DISTS:", dists.shape)
            fingertips_object_dist.append(torch.mean(dists, dim=-1).view(-1,1)) # n_robots * [agg_env,1]. avg distance over fingers
            # print("fingertips_object_dist:", torch.mean(dists, dim=-1).shape)
            # print("fingertips_object_dist:", torch.mean(dists, dim=-1).view(-1,1).shape)
        fingertips_object_dist = torch.cat(fingertips_object_dist, dim=1).reshape(-1) # [num_envs]
        fingertips_object_dist = torch.where(fingertips_object_dist >= 0.5, 0.5, fingertips_object_dist)

        flag = torch.logical_and((fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15), horizontal_offset <= max_horizontal_offset)
        flag_object_hand = torch.logical_or(fingertips_object_dist <= 0.12, palm_object_dist <= 0.15)

        # stage 1: lift reward 1
        object_goal_reward = torch.zeros_like(goal_object_dist)
        object_goal_reward = torch.where(flag >= 1, (0.9 - 2 * goal_object_dist), object_goal_reward)
        # stage 1: lift reward 2
        object_height = object_pos[:, 2]
        object_up = torch.zeros_like(goal_object_dist)
        object_up = torch.where(flag >= 1, 1 * (object_height - goal_height), object_up)
        
        # stage 2: lift to goal bonus
        bonus = torch.zeros_like(goal_object_dist)
        bonus = torch.where(flag >= 1, 
            torch.where(goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus), 
            bonus)

        # stage 3: success
        successes = torch.where(
            goal_object_dist <= success_tolerance,
            torch.where(
                flag >= 1, torch.ones_like(successes), successes
            ),
            torch.zeros_like(successes),
        )

        # penalty
        # object_fall = torch.where(
        #     (horizontal_offset>max_horizontal_offset)+(object_height<=0.3),
        #     torch.ones_like(task_successes),
        #     torch.zeros_like(task_successes)
        # )

        ### 9-8: forced lift test
        if test_forced_lift:
            # envs in the lifting stage: step count +1
            lift_step_count[:] = torch.where(is_lifting_stage>0, lift_step_count+1, lift_step_count) 
            # envs trigger the success criterion: go to lifting stage
            is_lifting_stage[:] = torch.where(successes>0, torch.ones_like(is_lifting_stage), is_lifting_stage) 
            # envs at the last step of lifting stage and hand-obj still close: task success
            task_successes = torch.zeros_like(goal_object_dist)
            task_successes = torch.where((lift_step_count>=n_lift_steps) & (flag_object_hand>=1), torch.ones_like(task_successes), task_successes) 
            # reset criterion
            resets = reset_buf.clone()
            resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
            resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
            resets = torch.where(lift_step_count>=n_lift_steps, torch.ones_like(resets), resets) # lifting stage end
        else:
            ### default / training case: hold several steps to accomplish the task
            cont_success_steps = torch.where(successes>0, cont_success_steps+1, cont_success_steps)
            task_successes = torch.zeros_like(goal_object_dist)
            task_successes = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(task_successes), task_successes)
            resets = reset_buf.clone()
            # print("reset:", resets)
            # print("progress_buf:", progress_buf)
            resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
            resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
            resets = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(resets), resets) # task success
        
        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(task_successes * resets.float())
        current_successes = torch.where(resets > 0, task_successes, current_successes)
        # print("current_successes:", current_successes)
        cons_successes = torch.where(
            num_resets > 0,
            av_factor * finished_cons_successes / num_resets
            + (1.0 - av_factor) * consecutive_successes,
            consecutive_successes,
        )

        # print("fingertip_obs_dist:", fingertips_object_dist)
        # print("palm_object_dist:", palm_object_dist)
        # print("object_goal_reward:", object_goal_reward)
        # print("object_up:", object_up)
        # reward shaping & information log
        reward = (
            - 2.0 * fingertips_object_dist
            - 1.0 * palm_object_dist
            + object_goal_reward
            + object_up
            + bonus
            + 200 * task_successes
            - 0.3 * horizontal_offset
            #- 100 * object_fall
        )

        info["fingertips_object_dist"] = fingertips_object_dist
        info["palm_object_dist"] = palm_object_dist
        info["object_goal_reward"] = object_goal_reward
        info["object_up"] = object_up
        info["bonus"] = bonus
        info["success_reward"] = task_successes
        info["horizontal_offset"] = horizontal_offset
        info["reward"] = reward
        info["hand_approach_flag"] = flag
        #info["object_fall"] = object_fall

        return (
            reward,
            resets,
            progress_buf,
            successes,
            current_successes,
            cons_successes,
            cont_success_steps,
            info,
        )
