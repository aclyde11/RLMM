"""
Unit and regression test for the rlmm package.
"""

import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
import rlmm.environment.openmmWrapper
import rlmm.environment.systemloader

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config

def test_load_test_system():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert (isinstance(env.step(0.2, 0.2, 0.2), tuple))


if __name__ == '__main__':
    test_load_test_system()
