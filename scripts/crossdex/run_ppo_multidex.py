# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from My_CrossDex_IsaacSim.tasks.direct.my_crossdex_isaacsim.my_crossdex_isaacsim_env_cfg import MyCrossdexIsaacsimEnvCfg
from My_CrossDex_IsaacSim.tasks.direct.my_crossdex_isaacsim.my_crossdex_isaacsim_env import MyCrossdexIsaacsimEnv

import algo

def build_runner(params, env):
    train_param = params

    if train_param["name"]=="ppo":
        from algo import ppo
        runner = ppo.PPO(
            vec_env = env,
            actor_critic_class = ppo.ActorCritic,
            train_param = train_param,
            apply_reset = True,
            is_vision = False,
        )
    else:
        raise ValueError("Unrecognized algorithm!")
    
    return runner

def get_train_params():
    params = {
        "name": "ppo",
        "log_dir": './runs_multidex' ,
        
        "policy":{ # only works for MlpPolicy right now
            "backbone_type": '',
            "freeze_backbone": False,
            "pi_hid_sizes": [1024, 1024, 512, 512],
            "vf_hid_sizes": [1024, 1024, 512, 512],
            "activation": "relu" # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        },
        "test": False,
        "resume": 0,
        # check for potential saves every this many iterations
        "save_interval": 500,# 500
        "print_log": True,

        # rollout params
        "max_iterations": 20000, # 20000

        # training params
        "cliprange": 0.2,
        "ent_coef": 0,
        "nsteps": 100,
        "noptepochs": 5,
        "nminibatches": 4,# this is per agent
        "max_grad_norm": 1,
        "optim_stepsize": 3.e-4,# 3e-4 is default for single agent training with constant schedule
        "schedule": "adaptive", # could be adaptive or linear or fixed
        "desired_kl": 0.016,
        "gamma": 0.96,
        "lam": 0.95,
        "init_noise_std": 0.8,
    }
    return params

def main():
    params = get_train_params()
    env_cfg = MyCrossdexIsaacsimEnvCfg()
    env = MyCrossdexIsaacsimEnv(env_cfg)
    obs = env.reset()
    
    runner = build_runner(params, env)

    while simulation_app.is_running():

        for i in range(1):
            runner.run()

        # for _ in range(env_cfg.scene.num_envs):
        #     actions = torch.zeros((env.action_space.shape), device="cuda")

        #     obs, reward, done, truncated, info = env.step(actions)

    print("end")

if __name__ == "__main__":
    # run the main function
    main()
    # while simulation_app.is_running(): continue
    # close sim app
    simulation_app.close()
    print("closed app")