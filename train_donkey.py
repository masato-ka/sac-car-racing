import os
import torch
import gym
import gym_donkeycar
import tensorflow as tf
from stable_baselines import SAC

from config import CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT, MIN_THROTTLE, MAX_THROTTLE, REWARD_CRASH
from env.vae_env import VaeEnv
from vae.vae import VAE

from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy

VARIANTS_SIZE = 32
DONKEY_SIM_PATH = f"/Applications/donkey_sim.app/Contents/MacOS/sdsim"
SIM_HOST="127.0.0.1"
DONKEY_SIM_PORT=9091
#DONKEY_SIM_PATH = f"remote"
#SIM_HOST = "trainmydonkey.com"

image_channels = 3

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[32, 16],
                                              act_fun=tf.nn.elu,
                                              feature_extraction="mlp")

def calc_reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
        return REWARD_CRASH - (CRASH_REWARD_WEIGHT * norm_throttle)
    throttle_reward = THROTTLE_REWARD_WEIGHT * (action[1] / MAX_THROTTLE)
    return 1 + throttle_reward

if __name__ == '__main__':

    model_path = 'vae-gt-80-160-18k-beta25-50-loss.torch'
    torch_device = 'cpu'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()
    #env = gym.make("donkey-warehouse-v0", exe_path=DONKEY_SIM_PATH, port=SIM_HOST)
    env = gym.make('donkey-generated-track-v0',exe_path=DONKEY_SIM_PATH, host=SIM_HOST, port=DONKEY_SIM_PORT)
    env.viewer.set_car_config("donkey", (128, 128, 128), "masato-ka", 20)
    vae_env = VaeEnv(env, vae, device=torch_device, reward_callback=calc_reward)

    '''
    Normal SAC in stable baselines but code is changed to calculate gradient only when done episode.
    In gym_donkey, skip_frame parameter is 2 but modify to 1. 
    '''
    model = SAC(CustomSACPolicy, vae_env, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300,
                gradient_steps=600, train_freq=1, ent_coef='auto_0.1', learning_rate=3e-4)
    model.learn(total_timesteps=30000, log_interval=1)
    model.save('donkey7')
