import importlib
import inspect
import yaml

from collections import namedtuple


ConfigBase = namedtuple('ConfigBase', ('env', 'systemloader', 'actions', 'obsmethods', 'openmmWrapper'))


class Config(ConfigBase):

    def __new__(cls, config_dict):
        configs = {}
        for k, v in config_dict.items():
            if k == 'env':
                envconfig = namedtuple('Environment', ('sim_steps', 'samples_per_step',
                                                       'movie_frames', 'verbose', 'tempdir'))
                configs['env'] = envconfig(**v)
                continue
            my_module = importlib.import_module('rlmm.environment.{}'.format(k))
            clsmembers = dict(inspect.getmembers(my_module, inspect.isclass))
            class_match = clsmembers[v['module']]
            configs[k] = class_match.Config(**v)
        return super().__new__(cls, **configs)


    # def __init__(self, config_dict):
    #     self.configs = {}
    #     for k, v in config_dict.items():
    #         if k == 'env':
    #             self.configs.update(v)
    #             continue
    #         my_module = importlib.import_module('rlmm.environment.{}'.format(k))
    #         clsmembers = dict(inspect.getmembers(my_module, inspect.isclass))
    #         class_match = clsmembers[v['module']]
    #         self.configs[k] = class_match.Config(v)


    def update(self, k, v):  # is this used? 
        self.__dict__.update({k : v})

    # # Load from yaml, alternative constructor
    @classmethod
    def load_yaml(cls, file):
        with open(file, 'r') as f:
            return cls(yaml.load(f, Loader=yaml.FullLoader))

    # Dump to yaml... maybe to be re-tooled into an 'accumulator' of sub-class dumps
    def dump_yaml(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(f)


if __name__ == "__main__":
    Config.load_yaml('test_config.yaml')
