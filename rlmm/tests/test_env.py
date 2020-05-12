"""
Unit and regression test for the rlmm package.
"""

# import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
# import rlmm.environment.openmmWrapper
# import rlmm.environment.systemloader

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from rlmm.rl.Expert import  ExpertPolicy,FastRocsPolicy, RandomPolicy
import pickle
def test_PDBLoader_get_mobile():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert(644 == env.systemloader.get_mobile())

def test_load_test_system():
    import logging
    import warnings
    import os
    import shutil
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)

    shutil.rmtree('rlmmtest')
    os.mkdir('rlmmtest')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs), sim_steps=100, samples_per_step=360, movie_frames=120)
    policy = FastRocsPolicy(env, hits=100, step_size=2.0, header='python /scratch/aclyde/fastrocs/ShapeDatabaseClient.py venti:8080')

    energies = []
    for i in range(100):
        choice = policy.choose_action()
        print("Action taken: ", choice[1])
        _, _, _, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)
        # print(data)




if __name__ == '__main__':
    test_load_test_system()
