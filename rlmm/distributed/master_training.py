import tempfile
import pickle
import numpy as np
from openeye import oechem, oedocking
from simtk import unit
from sklearn.decomposition import PCA
from rlmm.utils.loggers import make_message_writer


class MasterRandomPolicy:

    def __init__(self,
                 env,
                 env_config,
                 num_envs,
                 comm_type,
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

        if comm_type == 'MPI':
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
        pass

    def start_envs(self):
        """ spawns processes for each env and sets up envs on each process """
        pass

    def env_reset(self):
        """ calls reset() method for all envs and returns the array of obs like:
            [obs_from_env_0, obs_from_env_1, ...] """
        pass

    def env_step(self, choice_vector):
        """ performs a set on each env
            so sends choice_vector[0] to env0, choice_vector[1] to env1, and so on
            returns vectors of obs, rewards, dones, and data like:
            [obs0, ob1,...], [reward0, reward1,...], [done0, done1,...], [data0, data1,...]
        """
        pass

    def get_env_data(self):
        """ gets data from all envs like [env0.data, env1.data, ...] """
        pass

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
