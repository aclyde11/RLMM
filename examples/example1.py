from datetime import datetime
from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from rlmm.rl.Expert import  ExpertPolicy, RandomPolicy
import pickle
import os

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

    # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for k ,v in config.configs.items():
        if k in ['actions', 'systemloader', 'openmmWrapper', 'obsmethods']:
            for k_, v_ in config.configs.items():                                               # flattening?
                if k_ != k:
                    v.update(k_, v_)

def test_load_test_system():
    import logging
    import warnings
    import shutil
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CONFIG INIT
    config = Config.load_yaml('RLMM/examples/example1_config.yaml')                             # notes: idea > reformat yaml for pure dict load, currently loads nested lists, see yaml
    setup_temp_files(config)
    shutil.copy('RLMM/rlmm/tests/test_config.yaml', config.configs['tempdir'] + "config.yaml")  # save a copy of the config yaml

    # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ENV INIT
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))


    # import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> POLICY INIT
    policy = ExpertPolicy(env,num_returns=-1, sort='iscores', orig_pdb=config.configs['systemloader'].pdb_file_name)


    import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OBS INIT
    obs = env.reset()                                                                           # notes: obs init by environment with configurations
    energies = []

    import pdb; pdb.set_trace() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> START TEST
    for i in range(100):
        choice = policy.choose_action(obs)
        print("Action taken: ", choice[1])
        obs, reward, done, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)


if __name__ == '__main__':
    test_load_test_system()
