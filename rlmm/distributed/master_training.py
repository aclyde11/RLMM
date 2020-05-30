import tempfile
import pickle
import numpy as np
from openeye import oechem, oedocking
from simtk import unit
from sklearn.decomposition import PCA
from rlmm.utils.loggers import make_message_writer
from Communicators import MPICommunicator

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
