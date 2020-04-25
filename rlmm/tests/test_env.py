"""
Unit and regression test for the rlmm package.
"""

# Import package, test suite, and other packages as needed
import rlmm.environment.openmm
import pytest

def test_env_openmm_openmmenv_init():
    env = rlmm.environment.openmm.OpenMMEnv(1)
    assert env is not None
=======
import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
import rlmm.environment.openmmWrapper
import rlmm.environment.systemloader

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config

def test_PDBLoader_get_mobile():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert(644 == env.systemloader.get_mobile())

def test_load_test_system():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert (isinstance(env.step(0.2, 0.2, 0.2), tuple))


if __name__ == '__main__':
    test_PDBLoader_get_mobile()
 Add open mm example code (#2)
