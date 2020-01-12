import os
import torch
import gym
import gym_donkeycar
import tensorflow as tf
from stable_baselines import SAC

from config import CRASH_SPEED_WEIGHT, THROTTLE_REWARD_WEIGHT, MIN_SPEED, MAX_SPEED, REWARD_CRASH
from env.vae_env import VaeEnv
from vae.vae import VAE

from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy

VARIANTS_SIZE = 32
os.environ['DONKEY_SIM_PATH'] = f"/Applications/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0)
image_channels = 3

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[32, 16],
                                              act_fun=tf.nn.elu,
                                              feature_extraction="mlp")

def learning_callable(time):
    if time > 0.90:
        return 0.0005
    if time > 0.80:
        return 0.0003
    return 0.0001

def calc_reward(action, e_i, done):
    if done:
        norm_throttle = (action[1] - MIN_SPEED) / (MAX_SPEED- MIN_SPEED)
        return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
    throttle_reward = THROTTLE_REWARD_WEIGHT * (action[1] / MAX_SPEED)
    return 1 + throttle_reward

if __name__ == '__main__':

    model_path = 'vae-gr-100.torch'
    torch_device = 'cpu'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()

    env = gym.make('donkey-generated-roads-v0')
    vae_env = VaeEnv(env, vae, device=torch_device, reward_callback=calc_reward)

    model = SAC(CustomSACPolicy, vae_env, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300,
                gradient_steps=300, train_freq=1, ent_coef='auto_0.1', learning_rate=0.0003)
    model.learn(total_timesteps=10000, log_interval=1)
    model.save('donkey4')
