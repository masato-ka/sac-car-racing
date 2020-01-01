from unittest import TestCase

import gym

from env.vae_env import VaeEnv


class TestVaeEnv(TestCase):

    target = VaeEnv(gym.make('CarRacing-v0'), '')

    def test_reest(self):
        self.target.reset()
        #self.fail()
        pass
    def test_step(self):
        action = self.target.action_space
        self.target.step(action)

    def test_close(self):
        #self.fail()
        pass