import os
import pickle
import shutil
from datetime import datetime
import numpy as np

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.rl.Expert import ExpertPolicy
from rlmm.utils.config import Config


def setup_temp_files(config):
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        pass
    if config.configs['tempdir'][-1] != '/':
        config.configs['tempdir'] = config.configs['tempdir'] + "/"
    if not config.configs['overwrite_static']:
        config.configs['tempdir'] = config.configs['tempdir'] + "{}/".format(
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

    conf_file = 'examples/example1_config.yaml'
    config = Config.load_yaml(conf_file)
    setup_temp_files(config)
    shutil.copy(conf_file, config.configs['tempdir'] + "config.yaml")
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = ExpertPolicy(env, num_returns=-1, sort='dscores', orig_pdb=config.configs['systemloader'].pdb_file_name)
    obs_shapes = []
    obs = env.reset()
    with open("dim_logging.txt", "a+") as out:
        out.write("Reset obs:\n{}\n{}".format(type(obs), np.array(obs).shape))
        obs_shapes.append(np.array(obs).shape)
        energies = []
        for i in range(100):
            print("STEP ", i)
            out.write("STEP {}".format(i))
            choice = policy.choose_action(obs)
            #print("Action taken: ", choice[1])
            out.write("Choice {}:\n{}\n{}".format(i, type(choice), np.array(choice).shape))

            obs, reward, done, data = env.step(choice)
            out.write("Obs step {}:\n{}\n{}".format(i, type(obs), np.array(obs).shape))
            obs_shapes.append(np.array(obs).shape)
            out.write("Reward step {}:\n{}\n{}".format(i, type(reward), np.array(reward).shape))
            for k, v in data.items():
                out.write("{} step {}:\n{}\n{}".format(k, i, type(v), np.array(v).shape))

            energies.append(data['energies'])
            with open("rundata.pkl", 'wb') as f:
                pickle.dump(env.data, f)
            
    res1 = list(map(max, zip(*obs_shapes)))
    print("Max dim: ", res1)
    return obs_shapes


if __name__ == '__main__':
    test_load_test_system()

