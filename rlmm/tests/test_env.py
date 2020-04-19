"""
Unit and regression test for the rlmm package.
"""

# Import package, test suite, and other packages as needed
import rlmm.environment.openmm
import pytest

def test_env_openmm_openmmenv_init():
    env = rlmm.environment.openmm.OpenMMEnv(1)
    assert env is not None