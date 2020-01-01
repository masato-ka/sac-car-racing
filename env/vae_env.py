import PIL

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from gym import Env, spaces

from config import INPUT_DIM
from vae.vae import VAE


def normalize(x, amin=0, amax=1):
    xmax = x.max()
    xmin = x.min()
    if xmin == xmax:
        return np.ones_like(x)
    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin


class VaeEnv(Env):


    def __init__(self, wrapped_env, vae):
        super(VaeEnv, self).__init__()
        n_command_history = 0
        self._wrapped_env = wrapped_env
        self.vae = vae
        self.z_size = 15
        self.n_commands = 0
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.z_size + self.n_commands * n_command_history, ),
                                            dtype=np.float32)

#            spaces.Box(low=-20, high=20,shape=(15,), dtype=np.float32)
        self.action_space = wrapped_env.action_space

    def reset(self):
        observe = self._wrapped_env.reset()
        o = PIL.Image.fromarray(observe)
        o = o.resize((64,64), resample=PIL.Image.BICUBIC)
        tensor = transforms.ToTensor()(o)
        z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1])
        o = z.detach().cpu().numpy()[0]
        return o

    def step(self, action):
         observe, reward, done, e_i = self._wrapped_env.step(action)
         o = PIL.Image.fromarray(observe)
         o = o.resize((64,64), resample=PIL.Image.BICUBIC)
         tensor = transforms.ToTensor()(o)
         z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1])
         o = z.detach().cpu().numpy()[0]
         return o, reward, done, e_i

    def render(self):
        self._wrapped_env.render()

    def close(self):
        self._wrapped_env.close()


    def seed(self, seed=None):
        return self._wrapped_env.seed(seed)

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    def __str__(self):
        return super().__str__()



if __name__ == '__main__':
    import gym
    env = gym.make('CarRacing-v0')

    vae = VAE(image_channels=3, z_dim=15)
    vae.load_state_dict(torch.load('../vae.torch', map_location=torch.device('cpu')))
    vae.eval()

    vae_env = VaeEnv(env, vae)
    print(vae_env.observation_space.sample())
    vae_env.reset()
    action = vae_env.action_space.sample()
    o, r, d, e = vae_env.step(action)
    print(type(o))
    print('{} {} {} {}'.format(o,r,d,e))
    vae_env.close()
