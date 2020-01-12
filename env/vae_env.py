import time
from functools import wraps

import PIL

import torch

from torchvision import transforms
import numpy as np
from gym import Env, spaces


def normalize(x, amin=0, amax=1):
    xmax = x.max()
    xmin = x.min()
    if xmin == xmax:
        return np.ones_like(x)
    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin

MAX_STEERING_DIFF = 0.15
MAX_SPEED = 0.6
MIN_SPEED = 0.4
MAX_STEERING = 1
MIN_STEERING = -1
JERK_REWARD_WEIGHT = 0.0
DONE_SPEED_WEIGHT = 5
NORMAL_SPEED_WEIGHT = 0.1

def calc_reward(done, e_i, action):
    if done:
        # penalize the agent for getting off the road fast
        #print(e_i['speed'])
        norm_throttle = (action[1] - MIN_SPEED) / (MAX_SPEED- MIN_SPEED)
        return -10 - DONE_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
    throttle_reward = NORMAL_SPEED_WEIGHT * (action[1] / MAX_SPEED)
    return 1 + throttle_reward


class VaeEnv(Env):


    def __init__(self, wrapped_env, vae, device='cpu'):
        super(VaeEnv, self).__init__()
        self.device = torch.device(device)
        self._wrapped_env = wrapped_env
        self.vae = vae
        self.z_size = vae.z_dim
        self.n_commands = 2
        self.n_command_history = 10
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.z_size + (self.n_commands * self.n_command_history), ),
                                            dtype=np.float32)
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
#            spaces.Box(low=-20, high=20,shape=(15,), dtype=np.float32)
        self.action_space =spaces.Box(low=np.array([-MAX_STEERING, -1]),
                                      high=np.array([MAX_STEERING, 1]), dtype=np.float32)
            #wrapped_env.action_space

    def _record_action(self, action):

        if len(self.action_history) >= self.n_command_history * self.n_commands:
            del self.action_history[:2]
        for v in action:
            self.action_history.append(v)

    def _vae(self,observe):
        observe = PIL.Image.fromarray(observe)
        observe = observe.resize((64,64), resample=PIL.Image.BICUBIC)
        tensor = transforms.ToTensor()(observe)
        tensor.to(self.device)
        z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1].to(self.device))
        return z.detach().cpu().numpy()[0]

    def reset(self):
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        observe = self._wrapped_env.reset()
        #avoid zooming image
        #self._wrapped_env.env.t = 1.0
        #observe, _, _, _ = self._wrapped_env.step(self._wrapped_env.action_space.sample())
        o = self._vae(observe)
        if self.n_command_history > 0:
            o = np.concatenate([o, np.asarray(self.action_history)], 0)
        return o

    def step(self, action):
        #Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        action[1] = (1 - t) * 0.1 + 0.6 * t

        if self.n_command_history > 0:
            prev_steering = self.action_history[-2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        self._record_action(action)
        observe, reward, done, e_i = self._wrapped_env.step(action)

        if np.math.fabs(e_i['cte']) > 2:
            done = True
        else:
            done = False
        reward = calc_reward(done, e_i, action)
        o = self._vae(observe)

        if self.n_command_history > 0:
            o = np.concatenate([o, np.asarray(self.action_history)], 0)
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

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.action_history[0, -2 * (i + 1)]
                prev_steering = self.action_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def __str__(self):
        return super().__str__()

