import os
import pickle
import shutil
from datetime import datetime

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.rl.Expert import ExpertPolicy
from rlmm.utils.config import Config
from rlmm.utils.filecontext import FileContext


def setup_temp_files(config):
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        pass
    if config.configs['tempdir'][-1] == '/':
        config.configs['tempdir'] = config.configs['tempdir'][:-1]
    if not config.configs['overwrite_static']:
        config.configs['tempdir'] = config.configs['tempdir'] + "/{}".format(
            datetime.now().strftime("rlmm_%d_%m_%YT%H%M%S"))
        try:
            os.mkdir(config.configs['tempdir'])
        except FileExistsError:
            print("Somehow the directory already exists... exiting")
            exit()
    else:
        try:
            shutil.rmtree(config.configs['tempdir'])
            os.mkdir(config.configs['tempdir'])
        except FileExistsError:
            print("Somehow the directory already exists... exiting")
            exit()

    config.configs['tempdir'] = FileContext(simulation_workers=1, tmpdir=config.configs['tempdir'])

    for k, v in config.configs.items():
        if k in ['actions', 'systemloader', 'openmmWrapper', 'obsmethods']:
            for k_, v_ in config.configs.items():
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


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='expiriments/jak2/jak2_example_cuda.yaml')
    args = parser.parse_args()
    conf_file = args.c
    config = Config.load_yaml(conf_file)
    setup_temp_files(config)
    shutil.copy(conf_file, f"{config.configs['tempdir']()}/config.yaml")

    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = ExpertPolicy(env, num_returns=20, sort='dscores', trackHScores=False, orig_pdb=config.configs['systemloader'].pdb_file_name)

    obs = env.reset()
    energies = []
    for i in range(100):
        ### MASTER RANK
        env.config.tempdir.start_step(i+1)
        choice = policy.choose_action(obs)
        print("Action taken: ", choice[1])

        ## SLAVES RANK
        obs, reward, done, data = env.step(choice)
#        energies.append(data['energies'])
#        with open(config.configs['tempdir'] + "rundata.pkl", 'wb') as f:
#            pickle.dump(env.data, f)


if __name__ == '__main__':
    test_load_test_system()
