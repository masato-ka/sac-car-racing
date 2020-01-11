import os
import torch
import gym
import gym_donkeycar
import tensorflow as tf
from stable_baselines import SAC

from env.vae_env import VaeEnv
from sac_net.custom_sac import SACWithVAE
from vae.vae import VAE

from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.sac.policies import MlpPolicy
#from stable_baselines import SAC

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

if __name__ == '__main__':

    model_path = 'vae-gr-100.torch'
    torch_device = 'cpu'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()
    env = gym.make('donkey-generated-roads-v0')
    vae_env = VaeEnv(env, vae, device=torch_device)
    #model = SAC(CustomSACPolicy, vae_env, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300, ent_coef='auto_0.1', gradient_steps=600, train_freq=3000, learning_rate=0.0003)
    model = SAC(CustomSACPolicy, vae_env, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300,
                gradient_steps=300, train_freq=1, ent_coef='auto_0.1')
    # model = SACWithVAE(CustomSACPolicy, vae_env, verbose=1, batch_size=64, buffer_size=30000, learning_starts=300,
    #             gradient_steps=300, train_freq=6000, ent_coef='auto_0.1')

    model.learn(total_timesteps=20000, log_interval=1)
    model.save('donkey4')
