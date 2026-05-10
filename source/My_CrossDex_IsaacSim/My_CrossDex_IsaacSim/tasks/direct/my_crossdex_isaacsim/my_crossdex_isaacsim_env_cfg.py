# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from isaaclab.sim.utils import create_prim

import gymnasium.spaces as gym_spaces  
import numpy as np                  
@configclass
class MyCrossdexIsaacsimEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 100.0
    num_envs = 40#8192 # each hand

    for i in range(int(num_envs/4)):
        create_prim("/World/envs/leap_hand/env_" + str(i))
        create_prim("/World/envs/shadow_hand/env_" + str(i))
        # create_prim("/World/envs/allegro_hand/env_0")
        create_prim("/World/envs/inspire_hand/env_" + str(i))
        create_prim("/World/envs/schunk_svh_hand/env_" + str(i))

    # # - spaces definition
    num_eigengrasp_actions = 9
    arm_dof = 6
    num_actions = num_eigengrasp_actions + arm_dof
    # num_actions = 22  
    num_observations = 37
    # num_observations = _
    num_states = num_observations
    state_space = gym_spaces.Box(np.ones(num_states) * -np.Inf, np.ones(num_states) * np.Inf)
    action_space = gym_spaces.Box(np.ones(num_actions) * -1., np.ones(num_actions) * 1.)
    observation_space = gym_spaces.Box(np.ones(num_observations) * -np.Inf, np.ones(num_observations) * np.Inf)                                                                     
    
    clipObservations = 5
    clipActions = 1
    success_tolerance = 0.05

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    sim.physx.gpu_max_rigid_patch_count = 524288
    device = "cuda" 
    control_freq_inv = 1 
    render_fps: int = -1

    # rl
    reward_function = "v2"
    max_episode_length = 100
    reset_time = -1
    maxConsecutiveSuccesses = 2
    av_factor = 0.1
    goal_height = 1
    multi_task_label = "onehot"

    # hand_names = ["leap_hand"]
    # hand_names = ["shadow_hand"]
    # hand_names = ["allegro_hand"]
    # hand_names = ["inspire_hand"]
    # hand_names = ["schunk_svh_hand"]
    hand_names = ["leap_hand", "shadow_hand", "inspire_hand", "schunk_svh_hand"]
    num_robots = len(hand_names)
    #leap
    # arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    # hand_dof_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

    #shadow
    # arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    # hand_dof_names = ['FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
    
    #allegro
    # arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    # hand_dof_names = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0']

    # #inspire
    # arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    # hand_dof_names = ['index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint']

    # #svh
    # arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    # hand_dof_names = ['right_hand_Thumb_Opposition', 'right_hand_Thumb_Flexion', 'right_hand_j3', 'right_hand_j4', 'right_hand_index_spread', 'right_hand_Index_Finger_Proximal', 'right_hand_Index_Finger_Distal', 'right_hand_j14', 'right_hand_j5', 'right_hand_Finger_Spread', 'right_hand_Pinky', 'right_hand_j13', 'right_hand_j17', 'right_hand_ring_spread', 'right_hand_Ring_Finger', 'right_hand_j12', 'right_hand_j16', 'right_hand_Middle_Finger_Proximal', 'right_hand_Middle_Finger_Distal', 'right_hand_j15']

    # multidex
    arm_dof_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
    hand_dof_names = [["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
                      ['FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1'],
                      ['index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint'],
                      ['right_hand_Thumb_Opposition', 'right_hand_Thumb_Flexion', 'right_hand_j3', 'right_hand_j4', 'right_hand_index_spread', 'right_hand_Index_Finger_Proximal', 'right_hand_Index_Finger_Distal', 'right_hand_j14', 'right_hand_j5', 'right_hand_Finger_Spread', 'right_hand_Pinky', 'right_hand_j13', 'right_hand_j17', 'right_hand_ring_spread', 'right_hand_Ring_Finger', 'right_hand_j12', 'right_hand_j16', 'right_hand_Middle_Finger_Proximal', 'right_hand_Middle_Finger_Distal', 'right_hand_j15']]
    
    fingertip_links = [["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"],
                       ["thtip", "fftip", "mftip", "rftip", "lftip"],
                       ["thumb_tip",  "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
                       ["thtip", "fftip", "mftip", "rftip", "lftip"]]
    palm_links = ["base_link",
                  "base_link",
                  "arm_link6",
                  "arm_link6"]
    
    palm_offset = [0.0, 0.0, 0.05]
    
    num_hand_dofs = [len(hand_dof_names[0]), len(hand_dof_names[1]), len(hand_dof_names[2]), len(hand_dof_names[3])]
    print("********************************hand_dof:", num_hand_dofs)
    # num_hand_dofs = [len(hand_dof_names[i]) for i in range(num_robots)]
    num_arm_dofs = [len(arm_dof_names)]
    num_fingers = [len(fingertip_links[0]), len(fingertip_links[1]), len(fingertip_links[2]), len(fingertip_links[3])] # type: ignore
    agg_num_fingers = np.cumsum([0]+num_fingers[:-1])
    # agg_num_envs = [num_envs]
    # num_hand_dofs = [num_hand_dof]
    # num_arm_dofs = [num_arm_dof]
    dataset = "grab"
    add_random_dataset = False
    # num_eigengrasp_actions = 9
    num_rl_actions = num_eigengrasp_actions + arm_dof

    robot_idx = 0

    # object_root_pose = [[-1.0,0.0,0.5,1,0,0,0]]
    mustard_start_pose = [0.5, 0, 0, 0, 0, 0, 0]
    apple_start_pose = [0.5, 0, 0, np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0]
    tennis_ball_start_pose = [0.5, 0, 0, 0, 0, 0, 0]
    rubiks_cube_start_pose = [0.5, 0, 0, 0, 0, 0, 0]
    orange_start_pose = [0.5, 0, 0, 0, 0, 0, 0]
    mug_start_pose = [0.5, 0, 0, 0, 0, 0, 0]
    object_start_poses = [mustard_start_pose, apple_start_pose, tennis_ball_start_pose, rubiks_cube_start_pose, orange_start_pose, mug_start_pose]

    mustard_init_state = np.array(mustard_start_pose).astype(np.float32)
    apple_init_state = np.array(apple_start_pose).astype(np.float32)
    tennis_ball_init_state = np.array(tennis_ball_start_pose).astype(np.float32)
    rubiks_cube_init_state = np.array(rubiks_cube_start_pose).astype(np.float32)
    orange_init_state = np.array(orange_start_pose).astype(np.float32)
    mug_init_state = np.array(mug_start_pose).astype(np.float32)
    object_init_states = [mustard_init_state, apple_init_state, tennis_ball_init_state, rubiks_cube_init_state, orange_init_state, mug_init_state]
    object_init_state = mug_init_state

    obj_idx = 5

    # test
    test_forced_lift = False
    test = False
    n_lift_steps = 30
    
    #table 
#     table_cfg = RigidObjectCfg(
#     prim_path="/World/envs/env_.*/Table",
#     spawn=sim_utils.MeshCuboidCfg(
#         size=(1.5, 1.5, 0.1), # Dimensions (x, y, z)
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
# )
    
    # # robot(s)
    leap_cfg = ArticulationCfg(
        prim_path="/World/envs/leap_hand/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/robots/urdf_origin//rm65_leap_right/rm65_leap_right.usd",
        # fix_base=True,
        # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0,0.7071,0,0.7071), joint_pos={"arm_joint1": 0.0,
                                         "arm_joint2": 0.0,
                                         "arm_joint3": 0.0,
                                         "arm_joint4": 1.0,
                                         "arm_joint5": 0.0,
                                         "arm_joint6": 0.0,
                                         "a_0": 0.0,
                                         "a_1": 0.0,
                                         "a_2": 0.0,
                                         "a_3": 0.0,
                                         "a_4": 0.0,
                                         "a_5": 0.0,
                                         "a_6": 0.0,
                                         "a_7": 0.0,
                                         "a_8": 0.0,
                                         "a_9": 0.0,
                                         "a_10": 0.0,
                                         "a_11": 0.0,
                                         "a_12": 0.0,
                                         "a_13": 0.0,
                                         "a_14": 0.0,
                                         "a_15": 0.0,}
    ),
    actuators={
        "arm_joint1": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint1"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint2": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint2"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint3": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            # effort_limit_sim=30.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint4": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint4"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint5": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint5"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint6"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "a_0": ImplicitActuatorCfg(
            joint_names_expr=["a_0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),  
        "a_1": ImplicitActuatorCfg(
            joint_names_expr=["a_1"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_2": ImplicitActuatorCfg(
            joint_names_expr=["a_2"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_3": ImplicitActuatorCfg(
            joint_names_expr=["a_3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),      
        "a_4": ImplicitActuatorCfg(
            joint_names_expr=["a_4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_5": ImplicitActuatorCfg(
            joint_names_expr=["a_5"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_6": ImplicitActuatorCfg(
            joint_names_expr=["a_6"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_7": ImplicitActuatorCfg(
            joint_names_expr=["a_7"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_8": ImplicitActuatorCfg(
            joint_names_expr=["a_8"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_9": ImplicitActuatorCfg(
            joint_names_expr=["a_9"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),
        "a_10": ImplicitActuatorCfg(
            joint_names_expr=["a_10"],
            effort_limit_sim=0.95,          
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "a_11": ImplicitActuatorCfg(
            joint_names_expr=["a_11"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      
        "a_12": ImplicitActuatorCfg(
            joint_names_expr=["a_12"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      

        "a_13": ImplicitActuatorCfg(
            joint_names_expr=["a_13"],
            effort_limit_sim=0.95,                      
            stiffness=3.0,          
            damping=0.5,
            friction=0.01,
            armature=0.001           

        ),
        "a_14": ImplicitActuatorCfg(
            joint_names_expr=["a_14"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001       
        ),
        "a_15": ImplicitActuatorCfg(
            joint_names_expr=["a_15"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        )
    },
    )

    shadow_cfg = ArticulationCfg(
        prim_path="/World/envs/shadow_hand/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/robots/urdf_origin//rm65_shadow_right/rm65_shadow_right.usd",
        # fix_base=True,
        # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0,0.7071,0,0.7071), joint_pos={"arm_joint1": 0.0,
                                         "arm_joint2": 0.0,
                                         "arm_joint3": 0.0,
                                         "arm_joint4": 1.0,
                                         "arm_joint5": 0.0,
                                         "arm_joint6": 0.0,
                                         "FFJ4": 0.0,
                                         "LFJ5": 0.0,
                                         "MFJ4": 0.0,
                                         "RFJ4": 0.0,
                                         "THJ5": 0.0,
                                         "FFJ3": 0.0,
                                         "LFJ4": 0.0,
                                         "MFJ3": 0.0,
                                         "RFJ3": 0.0,
                                         "THJ4": 0.0,
                                         "FFJ2": 0.0,
                                         "LFJ3": 0.0,
                                         "MFJ2": 0.0,
                                         "RFJ2": 0.0,
                                         "THJ3": 0.0,
                                         "FFJ1": 0.0,
                                         "LFJ2": 0.0,
                                         "MFJ1": 0.0,
                                         "RFJ1": 0.0,
                                         "THJ2": 0.0,
                                         "LFJ1": 0.0,
                                         "THJ1": 0.0}
    ),
    actuators={
        "arm_joint1": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint1"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint2": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint2"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint3": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            # effort_limit_sim=30.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint4": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint4"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint5": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint5"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint6"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "FFJ4": ImplicitActuatorCfg(
            joint_names_expr=["FFJ4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),  
        "LFJ5": ImplicitActuatorCfg(
            joint_names_expr=["LFJ5"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "MFJ4": ImplicitActuatorCfg(
            joint_names_expr=["MFJ4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "RFJ4": ImplicitActuatorCfg(
            joint_names_expr=["RFJ4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),      
        "THJ5": ImplicitActuatorCfg(
            joint_names_expr=["THJ5"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "FFJ3": ImplicitActuatorCfg(
            joint_names_expr=["FFJ3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "LFJ4": ImplicitActuatorCfg(
            joint_names_expr=["LFJ4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "MFJ3": ImplicitActuatorCfg(
            joint_names_expr=["MFJ3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "RFJ3": ImplicitActuatorCfg(
            joint_names_expr=["RFJ3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "THJ4": ImplicitActuatorCfg(
            joint_names_expr=["THJ4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),
        "FFJ2": ImplicitActuatorCfg(
            joint_names_expr=["FFJ2"],
            effort_limit_sim=0.95,          
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "LFJ3": ImplicitActuatorCfg(
            joint_names_expr=["LFJ3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      
        "MFJ2": ImplicitActuatorCfg(
            joint_names_expr=["MFJ2"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      

        "RFJ2": ImplicitActuatorCfg(
            joint_names_expr=["RFJ2"],
            effort_limit_sim=0.95,                      
            stiffness=3.0,          
            damping=0.5,
            friction=0.01,
            armature=0.001           

        ),
        "THJ3": ImplicitActuatorCfg(
            joint_names_expr=["THJ3"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001       
        ),
        "FFJ1": ImplicitActuatorCfg(
            joint_names_expr=["FFJ1"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "LFJ2": ImplicitActuatorCfg(
            joint_names_expr=["LFJ2"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "MFJ1": ImplicitActuatorCfg(
            joint_names_expr=["MFJ1"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "RFJ1": ImplicitActuatorCfg(
            joint_names_expr=["RFJ1"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "THJ2": ImplicitActuatorCfg(
            joint_names_expr=["THJ2"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "LFJ1": ImplicitActuatorCfg(
            joint_names_expr=["LFJ1"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "THJ1": ImplicitActuatorCfg(
            joint_names_expr=["THJ1"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
    },
    )

    allegro_cfg = ArticulationCfg(
        prim_path="/World/envs/allegro_handnvs/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/robots/urdf_origin//rm65_allegro_right/rm65_allegro_right.usd",
        # fix_base=True,
        # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0,0.7071,0,0.7071), joint_pos={"arm_joint1": 0.0,
                                         "arm_joint2": 0.0,
                                         "arm_joint3": 0.0,
                                         "arm_joint4": 1.0,
                                         "arm_joint5": 0.0,
                                         "arm_joint6": 0.0,
                                         'joint_0.0': 0.0,
                                         "joint_1.0": 0.0,
                                         "joint_2.0": 0.0,
                                         "joint_3.0": 0.0,
                                         "joint_4.0": 0.0,
                                         "joint_5.0": 0.0,
                                         "joint_6.0": 0.0,
                                         "joint_7.0": 0.0,
                                         "joint_8.0": 0.0,
                                         "joint_9.0": 0.0,
                                         "joint_10.0": 0.0,
                                         "joint_11.0": 0.0,
                                         "joint_12.0": 0.0,
                                         "joint_13.0": 0.0,
                                         "joint_14.0": 0.0,
                                         "joint_15.0": 0.0,}
    ),
    actuators={
        "arm_joint1": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint1"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint2": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint2"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint3": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            # effort_limit_sim=30.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint4": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint4"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint5": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint5"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint6"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "joint_0.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_0.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),  
        "joint_1.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_1.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_2.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_2.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_3.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_3.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),      
        "joint_4.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_4.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_5.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_5.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_6.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_6.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_7.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_7.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_8.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_8.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_9.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_9.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),
        "joint_10.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_10.0"],
            effort_limit_sim=0.95,          
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "joint_11.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_11.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      
        "joint_12.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_12.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      

        "joint_13.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_13.0"],
            effort_limit_sim=0.95,                      
            stiffness=3.0,          
            damping=0.5,
            friction=0.01,
            armature=0.001           

        ),
        "joint_14.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_14.0"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001       
        ),
        "joint_15.0": ImplicitActuatorCfg(
            joint_names_expr=["joint_15.0"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
       
        
    },
    )

    inspire_cfg = ArticulationCfg(
        prim_path="/World/envs/inspire_hand/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/robots/urdf_origin//rm65_inspire_right/rm65_inspire_right.usd",
        # fix_base=True,
        # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0,0.7071,0,0.7071), joint_pos={"arm_joint1": 0.0,
                                         "arm_joint2": 0.0,
                                         "arm_joint3": 0.0,
                                         "arm_joint4": 1.0,
                                         "arm_joint5": 0.0,
                                         "arm_joint6": 0.0,
                                         "index_proximal_joint": 0.0,
                                         "middle_proximal_joint": 0.0,
                                         "pinky_proximal_joint": 0.0,
                                         "ring_proximal_joint": 0.0,
                                         "thumb_proximal_yaw_joint": 0.0,
                                         "index_intermediate_joint": 0.0,
                                         "middle_intermediate_joint": 0.0,
                                         "pinky_intermediate_joint": 0.0,
                                         "ring_intermediate_joint": 0.0,
                                         "thumb_proximal_pitch_joint": 0.0,
                                         "thumb_intermediate_joint": 0.0,
                                         "thumb_distal_joint": 0.0,}
    ),
    actuators={
        "arm_joint1": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint1"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint2": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint2"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint3": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            # effort_limit_sim=30.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint4": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint4"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint5": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint5"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint6"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),  
        "index_proximal_joint": ImplicitActuatorCfg(
            joint_names_expr=["index_proximal_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "middle_proximal_joint": ImplicitActuatorCfg(
            joint_names_expr=["middle_proximal_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "pinky_proximal_joint": ImplicitActuatorCfg(
            joint_names_expr=["pinky_proximal_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),      
        "ring_proximal_joint": ImplicitActuatorCfg(
            joint_names_expr=["ring_proximal_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "thumb_proximal_yaw_joint": ImplicitActuatorCfg(
            joint_names_expr=["thumb_proximal_yaw_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "index_intermediate_joint": ImplicitActuatorCfg(
            joint_names_expr=["index_intermediate_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "middle_intermediate_joint": ImplicitActuatorCfg(
            joint_names_expr=["middle_intermediate_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "pinky_intermediate_joint": ImplicitActuatorCfg(
            joint_names_expr=["pinky_intermediate_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "ring_intermediate_joint": ImplicitActuatorCfg(
            joint_names_expr=["ring_intermediate_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),
        "thumb_proximal_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["thumb_proximal_pitch_joint"],
            effort_limit_sim=0.95,          
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "thumb_intermediate_joint": ImplicitActuatorCfg(
            joint_names_expr=["thumb_intermediate_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),  
        "thumb_distal_joint": ImplicitActuatorCfg(
            joint_names_expr=["thumb_distal_joint"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                     
                         

        
    },
    )

    svh_cfg = ArticulationCfg(
        prim_path="/World/envs/schunk_svh_hand/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/robots/urdf_origin//rm65_svh_right/rm65_svh_right.usd",
        # fix_base=True,
        # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0,0.7071,0,0.7071), joint_pos={"arm_joint1": 0.0,
                                         "arm_joint2": 0.0,
                                         "arm_joint3": 0.0,
                                         "arm_joint4": 1.0,
                                         "arm_joint5": 0.0,
                                         "arm_joint6": 0.0,
                                         "right_hand_Middle_Finger_Proximal": 0.0,
                                         "right_hand_Thumb_Opposition": 0.0,
                                         "right_hand_index_spread": 0.0,
                                         "right_hand_j5": 0.0,
                                         "right_hand_Middle_Finger_Distal": 0.0,
                                         "right_hand_Thumb_Flexion": 0.0,
                                         "right_hand_Index_Finger_Proximal": 0.0,
                                         "right_hand_Finger_Spread": 0.0,
                                         "right_hand_ring_spread": 0.0,
                                         "right_hand_j15": 0.0,
                                         "right_hand_j3": 0.0,
                                         "right_hand_Index_Finger_Distal": 0.0,
                                         "right_hand_Pinky": 0.0,
                                         "right_hand_Ring_Finger": 0.0,
                                         "right_hand_j4": 0.0,
                                         "right_hand_j14": 0.0,
                                         "right_hand_j13": 0.0,
                                         "right_hand_j12": 0.0,
                                         "right_hand_j17": 0.0,
                                         "right_hand_j16": 0.0,}
    ),
    actuators={
        "arm_joint1": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint1"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint2": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint2"],
            # effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint3": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            # effort_limit_sim=30.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),

        "arm_joint4": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint4"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint5": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint5"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "arm_joint6": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint6"],
            # effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=20.0,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_Middle_Finger_Proximal": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Middle_Finger_Proximal"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),  
        "right_hand_Thumb_Opposition": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Thumb_Opposition"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_index_spread": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_index_spread"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_j5": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j5"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),      
        "right_hand_Middle_Finger_Distal": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Middle_Finger_Distal"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_Thumb_Flexion": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Thumb_Flexion"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_Index_Finger_Proximal": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Index_Finger_Proximal"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_Finger_Spread": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Finger_Spread"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_ring_spread": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_ring_spread"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_j15": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j15"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),
        "right_hand_j3": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j3"],
            effort_limit_sim=0.95,          
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_Index_Finger_Distal": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Index_Finger_Distal"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      
        "right_hand_Pinky": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Pinky"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      

        "right_hand_Ring_Finger": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_Ring_Finger"],
            effort_limit_sim=0.95,                      
            stiffness=3.0,          
            damping=0.5,
            friction=0.01,
            armature=0.001           

        ),
        "right_hand_j4": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j4"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001       
        ),
        "right_hand_j14": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j14"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        ),
        "right_hand_j13": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j13"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001                   
        ),                      

        "right_hand_j12": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j12"],
            effort_limit_sim=0.95,                      
            stiffness=3.0,          
            damping=0.5,
            friction=0.01,
            armature=0.001           

        ),
        "right_hand_j17": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j17"],
            effort_limit_sim=0.95,
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001       
        ),
        "right_hand_j16": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_j16"],
            effort_limit_sim=0.95,                  
            stiffness=3.0,
            damping=0.5,
            friction=0.01,
            armature=0.001
        )
    },
    )

    robot_assets_cfg = [leap_cfg, shadow_cfg, inspire_cfg, svh_cfg]

    # objects
    object_assets_cfg = []
    mustard_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/006_mustard_bottle/006_mustard_bottle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(mustard_start_pose[0], mustard_start_pose[1], mustard_start_pose[2]),
            rot=(mustard_start_pose[3], mustard_start_pose[4], mustard_start_pose[5], mustard_start_pose[6])
        ),
    )
    apple_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/apple/apple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(apple_start_pose[0], apple_start_pose[1], apple_start_pose[2]),
            rot=(apple_start_pose[3], apple_start_pose[4], apple_start_pose[5], apple_start_pose[6])
        ),
    )
    tennis_ball_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/tennis_ball/tennis_ball.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(tennis_ball_start_pose[0], tennis_ball_start_pose[1], tennis_ball_start_pose[2]),
            rot=(tennis_ball_start_pose[3], tennis_ball_start_pose[4], tennis_ball_start_pose[5], tennis_ball_start_pose[6])
        ),
    )
    rubiks_cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/rubiks_cube/rubiks_cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(rubiks_cube_start_pose[0], rubiks_cube_start_pose[1], rubiks_cube_start_pose[2]),
            rot=(rubiks_cube_start_pose[3], rubiks_cube_start_pose[4], rubiks_cube_start_pose[5], rubiks_cube_start_pose[6])
        ),
    )
    orange_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/orange/orange.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(orange_start_pose[0], orange_start_pose[1], orange_start_pose[2]),
            rot=(orange_start_pose[3], orange_start_pose[4], orange_start_pose[5], orange_start_pose[6])
        ),
    )
    mug_leap_cfg = RigidObjectCfg(
        prim_path="/World/envs/leap_hand/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/mug/mug.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,   # kilograms
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False,
        
        ),
        
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(mug_start_pose[0], mug_start_pose[1], mug_start_pose[2]),
            rot=(mug_start_pose[3], mug_start_pose[4], mug_start_pose[5], mug_start_pose[6])
        ),
    )
    mug_shadow_cfg = RigidObjectCfg(
        prim_path="/World/envs/shadow_hand/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/mug/mug.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,   # kilograms
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(mug_start_pose[0], mug_start_pose[1], mug_start_pose[2]),
            rot=(mug_start_pose[3], mug_start_pose[4], mug_start_pose[5], mug_start_pose[6])
        ),
    )
    mug_inspire_cfg = RigidObjectCfg(
        prim_path="/World/envs/inspire_hand/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/mug/mug.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,   # kilograms
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(mug_start_pose[0], mug_start_pose[1], mug_start_pose[2]),
            rot=(mug_start_pose[3], mug_start_pose[4], mug_start_pose[5], mug_start_pose[6])
        ),
    )
    mug_schunk_svh_cfg = RigidObjectCfg(
        prim_path="/World/envs/schunk_svh_hand/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/callab/Research/Fatemeh/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/source/My_CrossDex_IsaacSim/My_CrossDex_IsaacSim/tasks/direct/assets/objects/urdf/mug/mug.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,   # kilograms
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        articulation_enabled=False,
        fix_root_link=False
        ),
            # Optional: Set scale
            # joint_drive=sim_utils.schemas.JointDrivePropertiesCfg(stiffness=100, damping=10),
            scale=(1, 1, 1), 
            # fix_base=True,
            

        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(mug_start_pose[0], mug_start_pose[1], mug_start_pose[2]),
            rot=(mug_start_pose[3], mug_start_pose[4], mug_start_pose[5], mug_start_pose[6])
        ),
    )
    
    # object_assets_cfg = [mustard_cfg, apple_cfg, tennis_ball_cfg, rubiks_cube_cfg, orange_cfg, mug_cfg]
    object_assets_cfg = [mug_leap_cfg, mug_shadow_cfg, mug_inspire_cfg, mug_schunk_svh_cfg]
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=False)


