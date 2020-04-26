import random

import gym
import numpy as np
from gym import spaces

from rlmm.utils.config import Config

class Spaces(object):
    pass


class OpenMMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    class Config(Config):
        def __init__(self, configs):
            self.__dict__.update(configs)

    def __init__(self, config_: Config):
        """

        :param systemloader:
        """
        gym.Env.__init__(self)
        self.config = config_

        self.systemloader = self.config.systemloader.get_obj()

        self.obs_processor = self.config.obsmethods.get_obj()

        self.action = self.config.actions.get_obj()
        self.action_space = self.action.get_gym_space()
        self.observation_space = self.setup_observation_space()
        self.spaces = Spaces()
        self.spaces.action = self.action_space
        self.spaces.observation = self.observation_space
        self.spaces.action.shape = [self.action_space.n]
        self.spaces.observation.shape = self.observation_space.shape
        self.reset()

    def setup_action_space(self):
        """

        :return:
        """
        return spaces.Discrete(1)

    def setup_observation_space(self):
        """

        :return:
        """
        return spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8)

    def get_obs(self):
        """

        :return:
        """
        coords = self.openmm_simulation.get_coordinates()
        out = self.obs_processor(coords)
        return out

    def step(self, *action):
        """

        :param action:
        :return:
        """
        self.action.apply_action_simulation(action, self.openmm_simulation)
        self.openmm_simulation.run(1)

        obs = self.get_obs()

        return obs, \
               random.random(), \
               random.random(), \
               {}

    def reset(self):
        """

        :return:
        """
        self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)
        return self.get_obs()

    def render(self, mode='human', close=False):
        """

        :param mode:
        :param close:
        """
        pass

    def close(self):
        """

        """
        pass
