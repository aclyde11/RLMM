

class Config:
    def __init__(self):
        pass

class Configurable:
    def __init__(self, config_ : Config):
        if not isinstance(config_, dict):
            config_ = config_.__dict__
        [setattr(self, k,v) for k,v in filter(lambda t: "_" not in t[0][0], config_.items())]
