"""
Record teleoperation trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python collect_data.py
"""

import argparse
import os
from glob import glob

import numpy as np

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper

from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.devices import Keyboard


def collect_teleoperation_trajectory(env, timesteps=1000):
    """Run a teleoperation policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """
    # Keyboard params
    pos_sensitivity = 1.0
    rot_sensitivity = 1.0

    arm = "right"

    switch_on_grasp = True
    toggle_camera_on_grasp = False

    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    
    env.reset()
    dof = env.action_dim
    
    # Initialize teleoperation device (keyboard)
    device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    env.viewer.add_keypress_callback(device.on_press)

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    for t in range(timesteps):
        # Set active robot
        active_robot = env.robots[0]

        # Get the newest action
        action, grasp = input2action(
                device=device, robot=active_robot, active_arm=arm, env_configuration=None
            )
        
        # If action is none, then this a reset so we should break
        if action is None:
            break
        
        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested
        if last_grasp < 0 < grasp:
            if switch_on_grasp:
                arm = "left" if arm == "right" else "right"
            if toggle_camera_on_grasp:
                cam_id = (cam_id + 1) % num_cam
                env.viewer.set_camera(camera_id=cam_id)
        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if arm == "right":
                action = np.concatenate([action, rem_action])
            elif arm == "left":
                action = np.concatenate([rem_action, action])
            else:
                # Only right and left arms supported
                print(
                    "Error: Unsupported arm specified -- "
                    "must be either 'right' or 'left'! Got: {}".format(arm)
                )
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        env.step(action)
        env.render()
        if t % 100 == 0:
            print(t)


def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--directory", type=str, default="/tmp/")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # create original environment
    env = suite.make(
        **config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        render_camera="agentview",
        has_offscreen_renderer=False,
        control_freq=20,
    )
    data_directory = args.directory

    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, data_directory)

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # collect some data
    print("Collecting some random data...")
    collect_teleoperation_trajectory(env, timesteps=args.timesteps)

    # playback some data
    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")
    data_directory = env.ep_directory
    playback_trajectory(env, data_directory)