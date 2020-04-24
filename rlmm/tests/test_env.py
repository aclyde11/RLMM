"""
Unit and regression test for the rlmm package.
"""

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
    env.step([0.05, 0.05, 0.05])
    assert (env.openmm_simulation.get_pdb(file_name='rlmm/tests/test_out.pdb'))


if __name__ == '__main__':
    test_load_test_system()
