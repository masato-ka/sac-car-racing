import PIL

import torch
import torch.nn.functional as F
from gym.envs.box2d import CarRacing
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


    def __init__(self, wrapped_env, vae, device='cpu'):
        super(VaeEnv, self).__init__()
        isinstance(wrapped_env, CarRacing)
        n_command_history = 0
        self.device = torch.device(device)
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


    def _vae(self,observe):
        observe = PIL.Image.fromarray(observe)
        observe = observe.resize((64,64), resample=PIL.Image.BICUBIC)
        tensor = transforms.ToTensor()(observe)
        tensor.to(self.device)
        z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1].to(self.device))
        return z.detach().cpu().numpy()[0]

    def reset(self):
        observe = self._wrapped_env.reset()

        #avoid zooming image
        self._wrapped_env.env.t = 1.0
        observe, _, _, _ = self._wrapped_env.step(self._wrapped_env.action_space.sample())
        return self._vae(observe)

    def step(self, action):
         observe, reward, done, e_i = self._wrapped_env.step(action)
         o = self._vae(observe)
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

