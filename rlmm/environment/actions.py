from gym import spaces

from rlmm.utils.config import Config


class EuclidanActionSpace:
    class Config(Config):
        def __init__(self, configs):
            pass

        def get_obj(self):
            return EuclidanActionSpace(self)

    def __init__(self, config: Config):
        self.config = config

    def apply_action_simulation(self, action, simulation):
        action = (action[0],action[0],action[0]) # rlpyt gym wrapper actions are len 1, should be fixed later
        simulation.translate(*action)

    def get_gym_space(self):
        return spaces.Discrete(2)
