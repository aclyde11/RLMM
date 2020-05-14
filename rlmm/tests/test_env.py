"""
Unit and regression test for the rlmm package.
"""

# import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
# import rlmm.environment.openmmWrapper
# import rlmm.environment.systemloader
from datetime import datetime
from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from rlmm.rl.Expert import  ExpertPolicy, RandomPolicy
import pickle
import os


def test_PDBLoader_get_mobile():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert(644 == env.systemloader.get_mobile())

def setup_temp_files(config):
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        pass
    if config.configs['tempdir'][-1] != '/':
        config.configs['tempdir'] = config.configs['tempdir'] + "/"
    config.configs['tempdir'] = config.configs['tempdir'] + "{}/".format( datetime.now().strftime("rlmm_%d_%m_%YT%H%M%S"))
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        print("Somehow the directory already exists... exiting")
        exit()

    for k ,v in config.configs.items():
        if k in ['actions', 'systemloader', 'openmmWrapper', 'obsmethods']:
            for k_, v_ in config.configs.items():
                if k_ != k:
                    v.update(k_, v_)

    print("?",config.configs)

def test_load_test_system():
    import logging
    import warnings
    import shutil
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)



    # temp dir
    try:
        shutil.rmtree('rlmmtest')
    except FileNotFoundError:
        pass
    os.mkdir('rlmmtest')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")



    config = Config.load_yaml('RLMM/rlmm/tests/test_config.yaml')
    setup_temp_files(config)
    shutil.copy('RLMM/rlmm/tests/test_config.yaml', config.configs['tempdir'] + "config.yaml")
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = ExpertPolicy(env, num_returns=-1, orig_pdb=config.configs['systemloader'].pdb_file_name)

    first_obs = env.reset()
    energies = []

    import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    for i in range(100):
        choice = policy.choose_action(config.configs['systemloader'].pdb_file_name)
        print("Action taken: ", choice[1])
        _, _, _, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)
        # print(data)




if __name__ == '__main__':
    test_load_test_system()
