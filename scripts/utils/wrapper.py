from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def wrap_env(env, saved_vn=None):
    wrapped_env = Monitor(env)
    # wrapped_env = DummyVecEnv([lambda : wrapped_env])
    # if saved_vn is None:
    #     wrapped_env = VecNormalize(wrapped_env) 
    # else:
    #     wrapped_env = VecNormalize.load(saved_vn, wrapped_env)

    return wrapped_env