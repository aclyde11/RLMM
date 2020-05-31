

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


def run_from_yaml(yaml, steps=100):
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    logging.getLogger('openforcefield').setLevel(logging.WARNING)
    logging.getLogger('openmmtools').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

    conf_file = yaml
    config = Config.load_yaml(conf_file)
    setup_temp_files(config)
    shutil.copy(conf_file, f"{config.configs['tempdir']()}/config.yaml")

    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = get_policy(env, config.configs['policy'])

    obs, _, _, data = env.reset()
    for i in range(config.configs['general']['max_iters']):
        env.config.tempdir.start_step(i + 1)
        if data['flew_away']:
            obs = data['init_obs']

        choice = policy.choose_action(obs)
        print("Action taken: ", choice[1])

        obs, reward, done, data = env.step(choice)

    with open(config.configs['tempdir'] + "rundata.pkl", 'wb') as f:
        pickle.dump(env.data, f)


if __name__ == "__main__":
    import logging
    import os
    import pickle
    import shutil
    import warnings
    from datetime import datetime

    from openeye import oechem

    from rlmm.environment.openmmEnv import OpenMMEnv
    from rlmm.rl.Expert import ExpertPolicy, get_policy
    from rlmm.utils.config import Config
    from rlmm.utils.filecontext import FileContext

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='expiriments/jak2/jak2_example_cuda.yaml')
    args = parser.parse_args()

    run_from_yaml(args.c)
