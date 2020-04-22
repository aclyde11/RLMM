import random

import gym
import numpy as np
from gym import spaces

from rlmm.environment.openmmWrapper import OpenMMSimulationWrapper
from rlmm.environment.obsmethods import CoordinatePCA
from rlmm.environment.actions import EuclidanActionSpace
from rlmm.utils.config import Config, Configurable

class OpenMMEnv(gym.Env, Configurable):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    class Config(Config, Configurable):
        def __init__(self, args):
            Config.__init__(self)
            Configurable.__init__(self, args)

    def __init__(self, config_ : Config):
        """

        :param systemloader:
        """
        gym.Env.__init__(self)
        Configurable.__init__(self, config_)

        self.systemloader = self.system(self.system_config)

        self.obs_processor = self.obs(self.obs_config)

        self.action = self.action(self.action_config)
        self.action_space = self.action.get_gym_space()
        self.observation_space = self.setup_observation_space()

        self.reset()


    def setup_action_space(self):
        """

        :return:
        """
        return spaces.Discrete(2)

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
        print(out)
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
        self.openmm_simulation = self.openmm(self.openmm.Config({'systemloader' : self.systemloader}))
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
