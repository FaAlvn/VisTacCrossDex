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

def main():
    print("hello")
    env_cfg = MyCrossdexIsaacsimEnvCfg()
    env = MyCrossdexIsaacsimEnv(env_cfg)
    print((env.action_space.shape))
    obs, info = env.reset()
    while simulation_app.is_running():
        for _ in range(env_cfg.scene.num_envs):
            actions = torch.zeros((env.action_space.shape), device=env.device)

            obs, reward, done, truncated, info = env.step(actions)

            print(reward.mean())
    print("end")
if __name__ == "__main__":
    # run the main function
    main()
    # while simulation_app.is_running(): continue
    # close sim app
    simulation_app.close()
    print("closed app")