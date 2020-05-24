import tempfile
import sys
import pickle
import numpy as np
from openeye import oechem, oedocking
from simtk import unit
from sklearn.decomposition import PCA
from rlmm.utils.loggers import make_message_writer
from mpi4py import MPI

MASTER = 0

class MasterRandomPolicy:

    def __init__(self,
                 env,
                 env_config,
                 num_envs: int,
                 comm_type: str,
                 return_docked_pose=False,
                 num_returns=-1,
                 step_size=3.5,
                 train_steps=100):

        self.return_docked_pose = return_docked_pose
        self.num_returns = num_returns
        self.step_size = step_size
        self.train_steps = train_steps
        self.env = env
        self.env_config = env_config
        self.num_envs = num_envs
        self.comm_type = comm_type
        print("mrp made it here")
        if comm_type == 'MPI':
            #QUESTION - why do we pass env in here? what to do with it?
            self.communicator = MPICommunicator(env, env_config, num_envs)
        elif comm_type == 'TCP':
            #self.communicator = TCPCommunicator(env, env_config, num_envs)
            pass

        self.communicator.start_envs()

    def getscores(self, actions, gsmis, prot, num_returns=10, return_docked_pose=False):

        num_returns_vector = [(num_returns, len(actions[idx])-1)[num_returns <= 0] for idx in range(len(actions))]

        idxs = []
        for i in range(len(actions)):
            idxs_for_env = list(np.random.choice(len(actions[i]), min(num_returns_vector[i], len(actions[i]) - 1), replace=False).flatten())
            idxs.append(idxs_for_env)

        return self.communicator.get_scores_data(idxs, actions, gsmis)

    def choose_action(self, pdb_string):
        with tempfile.TemporaryDirectory() as dirname:
            with open("{}/test.pdb".format(dirname), 'w') as f:
                f.write(pdb_string)
            pdb = oechem.OEMol()
            prot = oechem.OEMol()
            lig = oechem.OEMol()
            wat = oechem.OEGraphMol()
            other = oechem.OEGraphMol()
            ifs = oechem.oemolistream("{}/test.pdb".format(dirname))
            oechem.OEReadMolecule(ifs, pdb)
            ifs.close()
            if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
                print("crap")
                exit()

            actions_vector, gsmis_vector = self.communicator.get_new_action_sets(aligner=lig)

            data_vector = self.getscores(actions_vector,
                                         gsmis_vector,
                                         prot,
                                         num_returns=self.num_returns,
                                         return_docked_pose=self.return_docked_pose)

            new_mol2_vector, action_vector = self.communicator.get_mol2_act_with_trials(data_vector, self.step_size, dirname)  # maybe dirname wount be passed, we need better logging
            self.communicator.apply_action_vector(new_mol2_vector, action_vector)

            return new_mol2_vector, action_vector

    def train(self):
        obs_vector = self.communicator.env_reset()
        energies = [[] for e in self.num_envs]
        for i in range(self.train_steps):

            choice_vector = self.choose_action(obs_vector)

            obs_vector, reward_vector, done_vector, data_vector = self.communicator.env_step(choice_vector)

            # record energies for each env
            for env_idx in range(self.num_envs):
                energies[env_idx].append(data_vector[env_idx]['energies'])

            # record data from all envs
            with open("rundata.pkl", 'wb') as f:
                pickle.dump(self.communicator.get_env_data(), f)


class MPICommunicator:

    def __init__(self, env, env_config, num_envs):
        """ initialize stuff for communication """
        #QUESTION - why do we pass env in here? what to do with it?
        self.initial_env = env
        self.env = None
        self.num_envs = num_envs
        self.config = env_config
        self.comm = MPI.COMM_WORLD
        print("comm:", self.comm)
        self.rank = self.comm.Get_rank()
        print("rank:", self.rank)
        self.world_size = self.comm.Get_size()
        print("world size:", self.world_size)
        self.envs = dict()

    def start_envs(self):
        """ spawns processes for each env and sets up envs on each process """
        #only workers (rank > 0) will set up an env
        if self.rank > 0:
            self.env = OpenMMEnv(OpenMMEnv.Config(self.config.configs))
            #send it back to the master to be added to master storage
            self.comm.send(self.env, dest=MASTER)
        else:#master
            #add the initial version of the env for each node to self.envs dict
            #QUESTION should we maintain the current state of the env?
            #perhaps storing this at creation is unnecessary
            for i in range(1, self.num_envs + 1):
                self.envs[i] = self.comm.recv(source=i)
        
        #sync up all nodes before continuing -- rather than just returning from first if block, for synchronization
        self.comm.Barrier()

    def env_reset(self):
        """ calls reset() method for all envs and returns the array of obs like:
            [obs_from_env_0, obs_from_env_1, ...] """
        if self.rank > 0:
            #TODO this need not be a dictionary because each node will only have its own environment
            obs = self.env.reset()
            self.comm.send(obs, dest=MASTER)#send objs to master
        else: #master
            #recv obs from all
            obs_array = list()
            #if we do this loop with threads, then will need to ensure appended in correct order
            for i in range(1, self.num_envs + 1):
                obs_array.append(self.comm.recv(source=i))
        
        self.comm.Barrier()
        if self.rank == MASTER:
            return obs_array

    def env_step(self, choice_vector):
        """ performs a set on each env
            so sends choice_vector[0] to env0, choice_vector[1] to env1, and so on
            returns vectors of obs, rewards, dones, and data like:
            [obs0, ob1,...], [reward0, reward1,...], [done0, done1,...], [data0, data1,...]
        """
        #QUESTION - does the master need to store the return vector for each env?
        #QUESTION - do we want to store the energies and write env data to file? like in ex2:
        """
        energies.append(data['energies'])
            with open( config.configs['tempdir'] + "rundata.pkl", 'wb') as f:
                pickle.dump(env.data, f)
        """
        if self.rank > 0:
            #choice vector is 0 indexed whereas rank is not
            obs, reward, done, data = self.env.step(choice_vector[self.rank -1])
            self.comm.send((obs, reward, done, data), dest=MASTER)
        else:#master
            obs = list()
            rewards = list()
            done = list()
            data = list()
            #this is done in serial so lists should maintain order
            #do we want to do this in threads?
            for i in range(1, self.num_envs + 1):
                _obs, _reward, _done, _data = self.comm.recv(source=i)
                obs.append(_obs)
                rewards.append(_reward)
                done.append(_done)
                data.append(_data)

        #sync all up at the end
        self.comm.Barrier()
        if self.rank == MASTER:
            return obs, rewards, done, data

    def get_env_data(self):
        """ gets data from all envs like [env0.data, env1.data, ...] """
        if self.rank > 0:
            #send data
            self.comm.send(self.env.data, dest=MASTER)
        else:#master
            env_data = list()
            for i in range(1, self.num_envs + 1):
                env_data.append(self.comm.recv(source=i))
        
        self.comm.Barrier()
        if self.rank == MASTER:
            return env_data

    def get_new_action_sets(self, aligner):
        """ calls get_new_action_set for each env with the given aligner ligand """
        pass

    def get_aligned_action_for_env(env_idx, action, gsmis):
        """ calls env.action.get_aligned_action(action, gsmis) for an environment specified by env_idx
        returns what env outputs """
        pass

    def get_scores_data(self, idxs_vector, actions_vector, gsmis_vector):
        """ gets aligned actions for given indexes for each environment """
        scores_data_vector = []
        # TODO this can be rewritten in a more efficient manner depending on parallelization that you implement
        # ideally this should be parallelized as much as possible, so you can change the loop and helper method get_aligned_action_for_env
        # i wrote this as an example
        for env_idx in len(idxs_vector):
            idxs = idxs_vector[env_idx]
            actions = actions_vector[env_idx]
            gsmis = gsmis_vector[env_idx]
            data = []

            # TODO we might want to rewrite try fail to be mpi-friendly, with better handling of fails in our parallel setup
            for idx in idxs:
                try:
                    new_mol, new_mol2, gs, action = self.get_aligned_action_for_env(env_idx, actions[idx], gsmis[idx])
                    data.append((new_mol, new_mol2, gs, action))
                except Exception as e:
                    print("get aligned action fail")
                    continue

            scores_data_vector.append(data)
        return scores_data_vector

    def apply_action_vector(self, new_mol2_vector, action_vector):
        """ applies action given new molecule to each environment by invoking action.apply_action for each env
            so something like env0.action.apply_action(new_mol2_vector[0], action_vector[0]), then env1.action.apply_action(new_mol2_vector[1], action_vector[1]) and so on
        """
        pass

    def get_mol2_act_with_trials(self, data_vector, step_size, dirname):
            """ reloads system on each environment until there is a successful run
                returns vector of new_mol2 and action for each env, like [new_mol2_env0, new_mol2_env1,...] and [action_env0, action_env1,...]
            """
            new_mol2_vector = []
            action_vector = []

            for env_idx in len(data_vector):
                not_worked = True
                idxs = list(range(len(data_vector[env_idx])))
                idx = idxs.pop(0)

                # TODO later once everything is working, we should write a more mpi-friendly try/except routine,
                # so in case everything fails in "mega fail" scenario, we properly shut down all mpi processes and exit or retry again (?)
                while not_worked:
                    try:
                        new_mol, new_mol2, gs, action = data_vector[env_idx][idx]
                        # TODO these of course would need to be written as calls to each env using mpi
                        self.env.systemloader.reload_system(gs, new_mol, "{}/test.pdb".format(dirname))
                        self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                           ln=self.env.systemloader,
                                                                                           stepSize=step_size * unit.femtoseconds,
                                                                                           prior_sim=self.env.openmm_simulation.simulation)
                        not_worked = False
                    except Exception as e:
                        print(e)
                        if len(idxs) == 0:
                            print("mega fail")
                            exit()
                        idx = idxs.pop(0)

                new_mol2_vector.append(new_mol2)
                action_vector.append(action)

            return new_mol2_vector, action_vector


#testing
if __name__ == "__main__":
    mrp = MasterRandomPolicy(None, None, 3, "MPI")
