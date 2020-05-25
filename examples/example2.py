import os
import pickle
import shutil
from datetime import datetime

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.rl.Expert import ExpertPolicy
from rlmm.utils.config import Config
from mpi4py import MPI


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

    bashCommand = "export AMBERHOME=/home/aclyde11/anaconda2/envs/rlmmdev"
    os.system(bashCommand)

    conf_file = 'examples/example2_config.yaml'
    config = Config.load_yaml(conf_file)
    setup_temp_files(config)
    shutil.copy(conf_file, config.configs['tempdir'] + "config.yaml")
    comm = MPI.COMM_WORLD
    print("-----comm:", comm)
    rank = comm.Get_rank()
    print("-----rank:", rank)
    env = OpenMMEnv(OpenMMEnv.Config(config.configs), rank=rank)
    #print("orig_pbd if None, the start_dobj.IsInitialized call will fail:", config.configs['systemloader'].pdb_file_name)

    obs = env.reset()
    energies = []

    
    world_size = comm.Get_size()
    print("world_size:", world_size)
    n = 3

    #for i in range(100):
    ### MASTER RANK
    if rank == 0:
        print("-----Rank: master; testing master_policy_setting")
        master(world_size, env, comm, obs, n, config)
    else:
        print("-----Rank: {}; testing master_policy_setting".format(rank))
        minon(comm, rank, env, energies, n, config)
    comm.Barrier()  


def master(world_size,
            env,
            comm,
            obs, 
            n,
            config,
            policy_setting="master_policy_setting"):
    policy = ExpertPolicy(env, num_returns=-1, sort='dscores', orig_pdb=config.configs['systemloader'].pdb_file_name)
    if policy_setting =="master_policy_setting":
        print("Running with master_policy_setting for {} steps".format(n))
        # We are trying to go 100 steps of training but not sure if this is correct
        cummulative_state = [[obs,0, False, False]]
        for i in range(n):
            print("--- on step: {}".format(n))
            # [obs,reward,done,data]            
            for m in range(1, world_size):
                if i == 0:
                    obs = cummulative_state[i][0]
                else:
                    obs = cummulative_state[i][0]
                choice = policy.choose_action(obs)
                comm.send(choice, dest=m)
                print("Master sent action {} to rank: {}".format(choice, m))

            states= []
            for j in range(1, world_size):
                received = comm.recv(source=j)
                states.append(received)
                print("received obj, reward, done, data of: {} from rank: {}".format(received, j))
            cummulative_state.append(states)
            print("~~~~~~~~~~~ CUMM_STATE: /n", len(cummulative_state), len(cummulative_state[1]), type(cummulative_state[1]))
    elif policy_setting== "rolling_policy_setting":
        # there should be a local policy deployed to each rank
        print("Running with rolling_policy_setting for {} steps".format(n))

        cummulative_state = []
        states= []

        for m in range(1, world_size):
            received = comm.recv(source=m)
            states.append(received)
            print("received obj, reward, done, data of: {} from rank: {}".format(received, j))
        cummulative_state.append(states)


def minon(comm,
        rank,
        env,
        energies,
        n,
        config, # think this should be a local policy; not sure how to structure. 
        policy_setting="master_policy_setting"):
    if policy_setting =="master_policy_setting":
        choice = comm.recv(source=0)
        print('Minon of rank: {} got action: {} from master'.format(rank,choice))
        # choice is an tuple of (new_mol, action)--> what do we want to do with new_mol?
        # from the step method, it looks like it expects a tuple, so I keep the 
        obs,reward, done, data = env.step(choice)
        energies.append(data['energies'])
        with open("rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)
        comm.send([obs,reward,done,data], dest=0)
        print( "Sending obj, reward, done, data of: {} to master".format([obs,reward,done,data]))

    elif policy_setting =="rolling_policy_setting":
        policy = ExpertPolicy(env, num_returns=-1, sort='dscores', orig_pdb=config.configs['systemloader'].pdb_file_name)

        rank_states = [[obs,0, False, False]]
        for i in range(n):
            choice = policy.choose_action(obs)
            print('Minon of rank: {} chose action: {}'.format(rank,choice))
            obs,reward, done, data = env.step(choice)
            energies.append(data['energies'])
            with open("rundata.pkl", 'wb') as f:
                pickle.dump(env.data, f)
            rank_states.append([obs,reward,done,data])
        comm.send(rank_states, dest=0)    
        print("Sending [obj, reward, done, data] for {} timesteps from rank: {} to master".format(n,rank))

# assume policy has a "train" or "update"
# We will be taking the define_policy flag, which specifies how policy is trained (master policy versus rollout)
# we need to implement both of those frameworks, but dont assume its a RL, deterministic, whatever policy. That is abstracted
# It would be nice to set param communication_type = tcp or mpi and then have it work

if __name__ == '__main__':
    test_load_test_system()

