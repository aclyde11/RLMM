import importlib
import inspect

import yaml


class Config:

    def __init__(self, config_dict):
        self.configs = {}

        for k, v in config_dict.items():
            if k == 'env':
                self.configs.update(v)
                continue
            my_module = importlib.import_module('rlmm.environment.{}'.format(k))
            clsmembers = inspect.getmembers(my_module, inspect.isclass)
            class_matches = (list(filter(lambda x: x[0] == v[0]['module'], clsmembers)))[0]
            self.configs[k] = class_matches[1].Config({k: v for d in v for k, v in d.items()})

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
