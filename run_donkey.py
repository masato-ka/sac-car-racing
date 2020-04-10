import os
import torch
import gym
import gym_donkeycar
import time
from env.vae_env import VaeEnv
from vae.vae import VAE


from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

VARIANTS_SIZE = 32
DONKEY_SIM_PATH = f"/Applications/donkey_sim.app/Contents/MacOS/sdsim"
SIM_HOST="127.0.0.1"
DONKEY_SIM_PORT=9091
image_channels = 3

if __name__ == '__main__':

    #model_path = 'vae-gt-80-160-10k-beta25-150.torch'#for 6
    #model_path = 'vae-gt-80-160-18k-beta25-50-loss.torch'
    model_path = 'vae-gt-30k-50.torch'
    torch_device = 'cpu'
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))
    vae.to(torch.device(torch_device))
    vae.eval()

    env = gym.make('donkey-generated-track-v0', exe_path=DONKEY_SIM_PATH, host=SIM_HOST, port=DONKEY_SIM_PORT)
    env.viewer.set_car_config("donkey", (128, 128, 128), "masato-ka", 20)
    vae_env = VaeEnv(env, vae, device=torch_device)

    model = SAC.load('donkey8')

    obs = vae_env.reset()
    dones=False
    for step in range(10000): # 500ステップ実行
        if step % 10 == 0: print("step: ", step)
        #if dones:
        #    o = env.reset()
        #    break
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vae_env.step(action)
#        env.render()
    env.close()