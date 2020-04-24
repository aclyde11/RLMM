
from rlmm.utils.config import Config




if __name__ == "__main__":
    try:
        Config.load_yaml('rlmm/tests/test_config_bad.yaml')
    except ValueError:
        pass
    else:
        raise ValueError('Fail: test_config_bad.yaml')