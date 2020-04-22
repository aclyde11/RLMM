"""
Unit and regression test for the rlmm package.
"""

import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
import rlmm.environment.openmmWrapper
import rlmm.environment.pdbutils
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.environment.pdbutils import AmberSystemLoader, PDBSystemLoader
from rlmm.environment.actions import EuclidanActionSpace
from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.environment.openmmWrapper import OpenMMSimulationWrapper
from rlmm.environment.obsmethods import CoordinatePCA

def test_env_openmm_openmmenv_init():
    env = rlmm.environment.openmmEnv.OpenMMEnv(1)
    assert env is not None

def setuprun():

    abs = {
        # 'system' : AmberSystemLoader,
        'system' : PDBSystemLoader,
        'openmm' : OpenMMSimulationWrapper,
        'action' : EuclidanActionSpace,
        'obs' : CoordinatePCA
    }
    confs = {}
    for k, v_ in abs.items():
        confs[k] = v_
        confs[k + "_config"] = v_.Config()

    return confs

def test_load_test_system():
    abs = setuprun()
    env = OpenMMEnv(OpenMMEnv.Config(abs))
    assert(isinstance(env.step(0.2, 0.2, 0.2), tuple))


if __name__ == '__main__':
    test_load_test_system()
