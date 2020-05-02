
from rlmm.utils.config import Config
from rlmm.utils.exceptions import BadConfigError


def test_config_load(f='rlmm/tests/config_yaml/test_config.yaml'):
    config = Config.load_yaml(f)
    for v in config.configs.values():
        assert isinstance(v, Config)

def test_bad_config_load(f='rlmm/tests/config_yaml/test_config_bad.yaml'):
    try:
        Config.load_yaml(f)
    except BadConfigError as b:
        pass
    else:
        raise BadConfigError('Bad Config File Handling!')


if __name__ == "__main__":
    test_config_load()
    test_bad_config_load()
