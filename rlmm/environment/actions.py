from gym import spaces

from rlmm.utils.config import Config


class EuclidanActionSpace:
    class Config(Config):
        def __init__(self, configs):
            self.ligand_only = configs['ligand_only']
            self.minimize = configs['minimize']

        def get_obj(self):
            return EuclidanActionSpace(self)

    def __init__(self, config: Config):
        self.config = config

    def apply_action_simulation(self, action, simulation):
        simulation.translate(*action, ligand_only=self.config.ligand_only, minimize=self.config.minimize)

    def get_gym_space(self):
        return spaces.Discrete(2)
