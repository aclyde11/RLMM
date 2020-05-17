from datetime import datetime
from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from rlmm.rl.Expert import  ExpertPolicy, RandomPolicy
import pickle
import os
from mpi4py import MPI


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
    mpi_logger = logging.getLogger("mpi_logs")
    mpi_logger.setLevel(logging.DEBUG)

    config = Config.load_yaml('examples/example1_config.yaml')
    setup_temp_files(config)
    shutil.copy('rlmm/tests/test_config.yaml', config.configs['tempdir'] + "config.yaml")
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = RandomPolicy(env)

    obs = env.reset()
    energies = []
    comm = MPI.COMM_WORLD
    print("comm:", comm)
    rank = comm.Get_rank()
    print("rank:", rank)
    world_size = comm.Get_size()
    print("world_size:", world_size)
    n = 100
    out = []
    if rank == 0:
        master(world_size, comm, obs, n, out, policy, mpi_logger)
    else:
        minon(comm, rank, env, energies, policy, mpi_logger)
    comm.Barrier()  
    mpi_logger.debug(out[-1])

def master(world_size, 
            comm,
            obs, 
            n,
            out,
            policy,
            mpi_logger,
            policy_setting="master_policy_setting"):
    if policy_setting =="master_policy_setting":
        mpi_logger.debug("Running with master_policy_setting for {} steps".format(n))
        # We are trying to go 100 steps of training but not sure if this is correct
        cummulative_state = [[obs,0, False, False]]
        for i in range(n):
            # [obs,reward,done,data]
            obs = cummulative_state[i][0]
            choice = policy.choose_action(obs)
            for m in range(1, world_size):
                comm.send(choice, dest=m)
                mpi_logger.debug("Master sent action {} to rank: {}".format(choice, m))

            states= []
            for j in range(1, world_size):
                received = comm.recv(source=j)
                states.append(received)
                mpi_logger.debug("received obj, reward, done, data of: {} from rank: {}".format(received, j))
            cummulative_state.append(states)
    
    elif policy_setting== "rolling_policy_setting":
        # there should be a local policy deployed to each rank
        mpi_logger.debug("Running with rolling_policy_setting for {} steps".format(n))
        cummulative_state = [[obs,0, False, False]]
        # [obs,reward,done,data]
        obs = cummulative_state[i][0]
        states= []

        for m in range(1, world_size):
            received = comm.recv(source=m)
            states.append(received)
            mpi_logger.debug("received obj, reward, done, data of: {} from rank: {}".format(received, j))
        cummulative_state.append(states)

    out.append(cummulative_state)


def minon(comm,
        rank,
        env,
        energies,
        policy, # think this should be a local policy; not sure how to structure. 
        mpi_logger,
        policy_setting="master_policy_setting"):
    if policy_setting =="master_policy_setting":
        choice = comm.recv(source=0)
        mpi_logger.debug('Minon of rank: {} got action: {} from master'.format(rank,choice))
        obs,reward, done, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)
        comm.send([obs,reward,done,data], dest=0)
        mpi_logger.debug("Sending obj, reward, done, data of: {} to master".format([obs,reward,done,data]))

    elif policy_setting =="rolling_policy_setting":
        choice = policy.choose_action(obs)
        mpi_logger.debug('Minon of rank: {} chose action: {}'.format(rank,choice))
        obs,reward, done, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)
        comm.send([obs,reward,done,data], dest=0)
        mpi_logger.debug("Sending obj, reward, done, data of: {} to master".format([obs,reward,done,data]))



# assume policy has a "train" or "update"
# We will be taking the define_policy flag, which specifies how policy is trained (master policy versus rollout)
# we need to implement both of those frameworks, but dont assume its a RL, deterministic, whatever policy. That is abstracted
# It would be nice to set param communication_type = tcp or mpi and then have it work

if __name__ == '__main__':
    test_load_test_system()