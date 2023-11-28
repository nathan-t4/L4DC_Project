import os

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from time import strftime
from argparse import ArgumentParser

from utils.wrapper import wrap_env

def train(args):
    assert args.env in suite.ALL_ENVIRONMENTS, "Invalid env flag!"
    assert args.controller in suite.ALL_CONTROLLERS, "Invalid controller flag!"
    assert args.robot in suite.ALL_ROBOTS, "Invalid robot flag!"

    if args.continue_training:
        assert args.env in args.dir, "Directory and environment mismatch"
        assert args.robot in args.dir, "Directory and robot mismatch"
        MODEL_PATH = args.dir
    else:
        MODEL_PATH = f'./exps/{args.env}-{args.robot}/{args.policy}/{strftime("%Y%m%d-%H%M%S")}/'
    
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
                has_renderer=False,
                has_offscreen_renderer=False,
                ignore_done=False,
                use_camera_obs=False,
                use_object_obs=True,
                control_freq=20,
                horizon=100,
                reward_shaping=True,
            )
    )
    eval_env = GymWrapper(
            suite.make(
                env_name=args.env,
                robots=args.robot,
                controller_configs=controller_config,
                has_renderer=False,
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
    eval_env = wrap_env(env)

    # Initialize sb3 RL policy
    if args.policy == 'SAC':
        if args.continue_training:
            model = SAC.load(path=MODEL_PATH,
                             env=env,
                             tensorboard_log=TB_PATH,
                    )
        else:
            model = SAC(policy='MlpPolicy',
                        env=env,
                        gamma=0.97,
                        tensorboard_log=TB_PATH,
                        verbose=1,
                    )
    elif args.policy == 'PPO':
        if args.continue_training:
            model = PPO.load(path=MODEL_PATH,
                             env=env,
                             tensorboard_log=TB_PATH,
                    )
        else:
            model = PPO(policy='MlpPolicy',
                        env=env,
                        gamma=0.97,
                        tensorboard_log=TB_PATH,
                        verbose=1,
                    )
    else:
        raise RuntimeError('Invalid policy flag!')
    
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
    parser.add_argument('--policy', type=str, default='SAC')
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--continue_training', action='store_true')
    args = parser.parse_args()

    train(args)