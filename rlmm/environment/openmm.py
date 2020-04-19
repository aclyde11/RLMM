import random
import gym
from gym import spaces
import numpy as np

class OpenMMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        super(OpenMMEnv, self).__init__()

        self.action_space = self.setup_action_space()
        self.observation_space = self.setup_observation_space()

    def setup_action_space(self):
        return spaces.Discrete(2)

    def setup_observation_space(self):
        return spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8)

    def get_obs(self):
        return np.random.rand((self.observation_space.shape))

    def step(self, action):
        obs = self.get_obs()
        info = {'randominfo' : random.random()}

        return obs, \
               random.random(), \
               random.random(), \
               info

    def reset(self):
        return self.get_obs()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass