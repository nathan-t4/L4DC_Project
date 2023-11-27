import os
from copy import deepcopy

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from time import strftime
from argparse import ArgumentParser

def train(args):
    assert args.env in suite.ALL_ENVIRONMENTS
    assert args.controller in suite.ALL_CONTROLLERS, "Invalid controller flag!"

    MODEL_PATH = f'./exps/{args.env}/{strftime("%Y%m%d-%H%M%S")}/'
    TB_PATH = os.path.join(MODEL_PATH, 'tb/')
    LOG_PATH = os.path.join(MODEL_PATH, 'log/')
    # Load controller config
    controller_config = load_controller_config(default_controller=args.controller)
    # Initialize robosuite training + evaluation environment
    env = GymWrapper(
            suite.make(
                env_name=args.env,
                robots=args.robot,
                controller_configs=controller_config,
                has_renderer=True,
                has_offscreen_renderer=False,
                ignore_done=False,
                use_camera_obs=False,
                control_freq=20,
                horizon=100,
            )
    )
    eval_env = Monitor(GymWrapper(
            suite.make(
                env_name=args.env,
                robots=args.robot,
                controller_configs=controller_config,
                has_renderer=True,
                has_offscreen_renderer=False,
                ignore_done=False,
                use_camera_obs=False,
                control_freq=20,
                horizon=100,
            )
        )
    )

    # Initialize sb3 RL policy (SAC for now)
    model = SAC(policy='MlpPolicy',
                env=env,
                gamma=0.97,
                tensorboard_log=TB_PATH,
                verbose=1,
            )
    
    # Callbacks
    # Evaluation callback
    eval_callback = EvalCallback(eval_env, 
                                eval_freq=5000,
                                n_eval_episodes=10,
                                deterministic=True,
                                log_path=LOG_PATH,
                                best_model_save_path=LOG_PATH,
                                verbose=1,
                                render=False)
    
    # Periodically save model callback
    auto_save_callback = CheckpointCallback(save_freq=5000,
                                            save_path=MODEL_PATH,
                                            name_prefix=f'{args.env}',
                                            save_replay_buffer=False,
                                            verbose=2)
    
    callbacks = [eval_callback, auto_save_callback]
        
    model.learn(total_timesteps=int(1e6),
                callback=callbacks,
                reset_num_timesteps=False)
    
    model.save(os.path.join(MODEL_PATH, f'{args.env}_final.zip'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, default='PickPlaceSingle')
    parser.add_argument('--controller', type=str, default='OSC_POSE')
    parser.add_argument('--robot', type=str, default='Panda')
    # parser.add_argument('--dir', type=str, default=None)
    # parser.add_argument('--policy', type=str, default='SAC')
    args = parser.parse_args()

    train(args)