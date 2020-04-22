import yaml

# Configurable sub-class
class Configurable_1:
    def __init__(self, options):
        print(options)
        pass

# Configuable sub-class
class Configurable_2:
    def __init__(self, options):
        print(options)
        pass

# Configurable container master-class
class Config:

    # Configurable sub-class constructor look-up
    configurable = {
        'Configurable_1': Configurable_1,
        'Configurable_2': Configurable_2
    }

    # Calls upon the configurable sub-class constructors
    # Create a hash-table of configurable sub-class objects
    def __init__(self, config_dict):
        self.configs = {}
        for k, v in config_dict.items():
            self.configs[k] = self.configurable[k](v)
        pass

    # Load from yaml, alternative constructor
    @classmethod
    def load_yaml(cls, file):
        with open(file, 'r') as f:
            cls(yaml.load(f, Loader=yaml.FullLoader))

    # Dump to yaml... maybe to be re-tooled into an 'accumulator' of sub-class dumps
    def dump_yaml(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(f)


if __name__ == "__main__":
    Config.load_yaml('test.config.yaml')


class Configurable:
    def __init__(self, config_ : Config):
        if not isinstance(config_, dict):
            config_ = config_.__dict__
        [setattr(self, k,v) for k,v in filter(lambda t: "_" not in t[0][0], config_.items())]
