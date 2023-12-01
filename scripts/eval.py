import os
from argparse import ArgumentParser

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor

from utils.wrapper import wrap_env

def rollout(env, model, eps, render=True, deterministic=True):
    ep = 0
    success = 0
    obs, info = env.reset()
    while ep < eps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if render:
            env.unwrapped.render()

        if done:
            ep += 1
            if env.unwrapped._check_success():
                success += 1
            obs, info = env.reset()
            
    env.close()

    print(f'Success rate: {success / eps * 100}%')

def eval(args):
    assert args.env in suite.ALL_ENVIRONMENTS, "Invalid env flag!"
    assert args.controller in suite.ALL_CONTROLLERS, "Invalid controller flag!"
    assert args.robot in suite.ALL_ROBOTS, "Invalid robot flag!"

    MODEL_PATH = args.dir
    controller_config = load_controller_config(default_controller=args.controller)

    # Check flags are consistent
    assert args.env in MODEL_PATH, "env / dir flags mismatch!"
    assert args.robot in MODEL_PATH, "robot / dir flags mismatch"
    assert args.controller in MODEL_PATH, "controller / dir flags mismatch!"
    assert args.policy in MODEL_PATH, "policy / dir flags mismatch!"

    if args.policy == 'SAC':
        model = SAC.load(MODEL_PATH)
    elif args.policy == 'PPO':
        model = PPO.load(MODEL_PATH)
    else:
        raise RuntimeError("Invalid policy flag!")

    # Create evaluation environment (same params as training env)
    env = GymWrapper(
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
    # Wrap environment
    env = wrap_env(env)

    rollout(env=env, model=model, eps=args.eval_eps)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--eval_eps', type=int, default=10)
    parser.add_argument('--controller', type=str, default='OSC_POSITION')
    parser.add_argument('--robot', type=str, default='Panda')
    parser.add_argument('--policy', type=str, default='SAC')
    args = parser.parse_args()

    eval(args)