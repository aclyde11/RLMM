import random

import gym
import numpy as np
from gym import spaces

from rlmm.utils.config import Config


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
        self.out_number = 0
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
        return out

    def step(self, action, sim_steps=10):
        """

        :param action:
        :return:
        """

        from rdkit import Chem
        from rdkit.Chem import AllChem
        actions, gsmis = self.action.get_new_action_set()
        idxs = list(np.random.choice(len(actions), 10).flatten())
        idx = idxs.pop(0)
        not_worked=True
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = self.action.get_aligned_action(actions[idx], gsmis[idx])
                self.openmm_simulation.get_pdb("test.pdb")
                self.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader, ln=self.systemloader)
                not_worked=False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)
        self.action.apply_action(new_mol2, action)

        self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
        self.out_number += 1
        for i in range(60):
            self.openmm_simulation.run(416) #4166
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
            self.out_number += 1

        obs = self.get_obs()

        return obs, \
               random.random(), \
               random.random(), \
               {}

    def reset(self):
        """

        :return:
        """
        self.action.setup(self.config.systemloader.ligand_file_name)
        self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)
        self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
        self.out_number += 1
        for i in range(90):
            self.openmm_simulation.run(277) #2777
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
            self.out_number += 1

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
