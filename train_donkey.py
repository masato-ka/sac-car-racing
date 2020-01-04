import os
import torch
import gym
import gym_donkeycar

from env.vae_env import VaeEnv
from vae.vae import VAE


from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

VARIANTS_SIZE = 15
os.environ['DONKEY_SIM_PATH'] = f"/Applications/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0)
image_channels = 3

if __name__ == '__main__':

    model_path = 'vae-donkey.torch'
    torch_device = 'cpu'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()

    env = gym.make('donkey-generated-track-v0')
    vae_env = VaeEnv(env, vae, device=torch_device)
    model = SAC(MlpPolicy, vae_env, verbose=1, batch_size=64, learning_starts=500)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save('donkey')
