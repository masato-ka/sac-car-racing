import torch

import gym

from env.vae_env import VaeEnv
from vae.vae import VAE

import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

VARIANTS_SIZE = 15
image_channels = 3

if __name__ == '__main__':

    model_path = 'vae.torch'

    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    vae.eval()

    env = gym.make('CarRacing-v0')
    vae_env = VaeEnv(env, vae)

    model = SAC(MlpPolicy, vae_env, verbose=1)
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save('car_racing')
