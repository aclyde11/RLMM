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
        #Question - do we need to return anything to caller?
        if self.rank > 0:
            mols, smiles = self.env.action.get_new_action_set(aligner) #should this be env.get or env.action.get ?
        
        self.comm.Barrier()

    #is this function only called by get_scores_data and no others?
    def get_aligned_action_for_env(env_idx, action, gsmis):
        """ calls env.action.get_aligned_action(action, gsmis) for an environment specified by env_idx
        returns what env outputs """
        #Question - do we need to return anything to caller?
        if self.rank > 0:
            return self.env.action.get_aligned_action(action, gsmis)
        
    #this one is interesting because it calls the above function
    #i think the way it should work is this is the highest level function so it deals with message passing
    def get_scores_data(self, idxs_vector, actions_vector, gsmis_vector):
        """ gets aligned actions for given indexes for each environment """
        scores_data_vector = []
        #QUESTION - does env_idx refer to rank? or should all of this be run by each worker?
        for env_idx in len(idxs_vector):
            idxs = idxs_vector[env_idx]
            actions = actions_vector[env_idx]
            gsmis = gsmis_vector[env_idx]
            data = [] #to be filled with lists of data for each node

            # TODO we might want to rewrite try fail to be mpi-friendly, with better handling of fails in our parallel setup
            for idx in idxs:
                iter_data = [] #contains data for each node
                if self.rank > 0:
                    try:
                        new_mol, new_mol2, gs, action = self.get_aligned_action_for_env(env_idx, actions[idx], gsmis[idx])
                        self.comm.send((new_mol, new_mol2, gs, action), dest=MASTER)
                        #data.append((new_mol, new_mol2, gs, action))
                    except Exception as e:
                        print("get aligned action fail")
                        #should we send back dummy data to prevent forever blocking on failure? like below
                        #self.comm.send(None, dest=MASTER)
                else:#master
                    #retrieve data from all environments for each iteration
                    for i in range(1, self.num_envs + 1):
                        #how does this handle an error on worker side? should we set a timeout so it doesn't block forever?
                        iter_data.append(self.comm.recv(source=i))

                self.comm.Barrier() #synchronize iterations
                if self.rank == MASTER:
                    data.append(iter_data)

            if self.rank == MASTER:
                scores_data_vector.append(data)

        if self.rank == MASTER:
            return scores_data_vector

    def apply_action_vector(self, new_mol2_vector, action_vector) -> None: #does this have a return value? 
        """ applies action given new molecule to each environment by invoking action.apply_action for each env
            so something like env0.action.apply_action(new_mol2_vector[0], action_vector[0]), then env1.action.apply_action(new_mol2_vector[1], action_vector[1]) and so on
        """
        if self.rank > 0:
            self.env.action.apply_action(new_mol2_vector[self.rank -1], action_vector[self.rank -1]) #recall that rank vectors are zero indexed
        
        self.comm.Barrier()

    def get_mol2_act_with_trials(self, data_vector, step_size, dirname):
            """ reloads system on each environment until there is a successful run
                returns vector of new_mol2 and action for each env, like [new_mol2_env0, new_mol2_env1,...] and [action_env0, action_env1,...]
            """
            new_mol2_vector = []
            action_vector = []

            for env_idx in len(data_vector):
                
                if self.rank > 0:
                    not_worked = True
                    idxs = list(range(len(data_vector[env_idx])))
                    idx = idxs.pop(0)
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
                            self.comm.send((new_mol2, action), dest=MASTER)

                        except Exception as e:
                            print(e)
                            #QUESTION - should we self.comm.send(None, dest=Master) so that doesn't hold up? Or do that after a certain numebr of retries?
                            if len(idxs) == 0:
                                #in case everything fails in "mega fail" scenario, we properly shut down all mpi processes and exit or retry again (?)
                                #i think calling exit will kill each individual process, since all this code runs for each worker node
                                print("mega fail")
                                exit()
                            idx = idxs.pop(0)
                
                else: #master
                    mol2_vec_from_workers = []
                    action_vec_from_workers = []
                    #await each env to send 
                    for i in range(1, self.num_envs + 1):
                        #recv from each env and appent to running list
                        new_mol2_wrk, action_wrk = self.comm.recv(source=i)
                        mol2_vec_from_workers.append(new_mol2_wrk)
                        action_vec_from_workers.append(action_wrk)

                    #append running list to output vectors
                    new_mol2_vector.append(new_mol2)
                    action_vector.append(action)
                
                self.comm.Barrier()

            return new_mol2_vector, action_vector


#testing
if __name__ == "__main__":
    mrp = MasterRandomPolicy(None, None, 3, "MPI")
