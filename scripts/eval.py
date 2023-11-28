import os
from argparse import ArgumentParser

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from utils.wrapper import wrap_env

def rollout(env, model, eps, render=True, deterministic=True):
    ep = 0
    obs, info = env.reset()
    while ep < eps:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        print(reward)
        if render:
            env.unwrapped.render()

        if done:
            obs, info = env.reset()
            ep += 1
    
    env.close()

def eval(args):
    assert args.env in suite.ALL_ENVIRONMENTS, "Invalid env flag!"
    assert args.controller in suite.ALL_CONTROLLERS, "Invalid controller flag!"
    assert args.robot in suite.ALL_ROBOTS, "Invalid robot flag!"

    MODEL_PATH = args.dir
    controller_config = load_controller_config(default_controller=args.controller)

    env = Monitor(
        GymWrapper(
            suite.make(
                env_name=args.env,
                robots=args.robot,
                controller_configs=controller_config,
                has_renderer=True,
                has_offscreen_renderer=False,
                ignore_done=False,
                use_camera_obs=False,
                use_object_obs=True,
                control_freq=20,
                horizon=100,
                reward_shaping=True,
            )   
        )  
    )
    
    model = SAC.load(MODEL_PATH)

    rollout(env=env, model=model, eps=args.eval_eps)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--eval_eps', type=int, default=10)
    parser.add_argument('--controller', type=str, default='OSC_POSE')
    parser.add_argument('--robot', type=str, default='Panda')
    args = parser.parse_args()

    eval(args)