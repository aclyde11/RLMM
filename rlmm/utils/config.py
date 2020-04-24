import importlib
import inspect

import yaml


class Config:
    config_heads = {'systemloader', 'openmmWrapper', 'obsmethods', 'actions'}

    def __init__(self, config_dict):
        self.configs = {}

        for k, v in config_dict.items():
            my_module = importlib.import_module(f'rlmm.environment.{k}')
            clsmembers = inspect.getmembers(my_module, inspect.isclass)
            class_matches = (list(filter(lambda x: x[0] == v[0]['module'], clsmembers)))[0]
            self.configs[k] = class_matches[1].Config({k: v for d in v for k, v in d.items()})

    # # Load from yaml, alternative constructor
    @classmethod
    def load_yaml(cls, file):
        with open(file, 'r') as f:
            l = yaml.load(f, Loader=yaml.FullLoader)
            if not cls.config_heads - l.keys():
                return cls(l)
            else: # else: // bad file
                raise ValueError(f'Bad File Exception -- Missing: {cls.config_heads - l.keys()}')


    # Dump to yaml... maybe to be re-tooled into an 'accumulator' of sub-class dumps
    def dump_yaml(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(f)





