import importlib
import inspect

import yaml


class Config:
    # notes: currently the load in from the example config yaml reads in nested lists w/ current formatting
    def __init__(self, config_dict):
        # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.configs = {}

        for k, v in config_dict.items():
            if k == 'env':
                self.configs.update(v)                                                         # environment configurations live in the config_dict top-level
                continue
            my_module = importlib.import_module('rlmm.environment.{}'.format(k))
            clsmembers = dict(inspect.getmembers(my_module, inspect.isclass))                  # modified: make a dict
            class_match = clsmembers[v['module']]                                              # modified: more readable?
            self.configs[k] = class_match.Config(v)                                            # notes: dict comp. for flattening, can this be more readable?

        # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



    def update(self, k, v):
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
