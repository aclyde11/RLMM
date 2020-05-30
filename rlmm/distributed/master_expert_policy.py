import tempfile
import pickle
import numpy as np
from openeye import oechem, oedocking
from simtk import unit
from sklearn.decomposition import PCA
from rlmm.utils.loggers import make_message_writer
from Communicators import MPICommunicator


class MasterExpertPolicy:

    def __init__(self,
                 env,
                 env_config,
                 num_envs,
                 comm_type,
                 sort='dscores',
                 return_docked_pose=False,
                 num_returns=-1,
                 orig_pdb=None,
                 use_hybrid=False,
                 track_hscores=True)

        self.sort = sort
        self.return_docked_pose = return_docked_pose
        self.num_returns = num_returns
        self.env = env
        self.env_config = env_config
        self.comm_type = comm_type
        self.num_envs = num_envs

        self.orig_pdb = orig_pdb
        self.start_dobj = None
        self.start_receptor = None
        self.track_hscores = track_hscores
        assert( not (not self.track_hscores and self.sort == 'hscores'))
        self.past_receptors = []
        self.past_dockobjs = []
        self.past_coordinates = []
        self.pca = PCA(2)
        self.dockmethod = oedocking.OEDockMethod_Hybrid if use_hybrid else oedocking.OEDockMethod_Chemgauss4

        if comm_type == 'MPI':
            self.communicator = MPICommunicator(env, env_config, num_envs)
        elif comm_type == 'TCP':
            #self.communicator = TCPCommunicator(env, env_config, num_envs)
            pass

        if self.orig_pdb is not None:
            pdb = oechem.OEMol()
            prot = oechem.OEMol()
            lig = oechem.OEMol()
            wat = oechem.OEGraphMol()
            other = oechem.OEGraphMol()
            ifs = oechem.oemolistream(self.orig_pdb)
            oechem.OEReadMolecule(ifs, pdb)
            ifs.close()
            if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
                #logger.failure("Could not split complex", exit_all=True)
                print("failed to split complex")

            self.start_receptor = oechem.OEGraphMol()
            #logger.log("Building initial receptor file...")
            oedocking.OEMakeReceptor(self.start_receptor, prot, lig)

            self.start_dobj = oedocking.OEDock(self.dockmethod)
            self.start_dobj.Initialize(self.start_receptor)
            assert (self.start_dobj.IsInitialized())


    def getscores(self, actions, gsmis, prot, lig, num_returns=10, return_docked_pose=False):
        num_returns_vector = [(num_returns, len(actions[idx])-1)[num_returns <= 0] for idx in range(len(actions))]

        idxs = []
        for i in range(len(actions)):
            idxs_for_env = list(np.random.choice(len(actions[i]), min(num_returns_vector[i], len(actions[i]) - 1), replace=False).flatten())
            idxs.append(idxs_for_env)

        protein = oechem.OEMol(prot)
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, lig)
        dockobj = oedocking.OEDock(self.dockmethod)
        dockobj.Initialize(receptor)
        assert (dockobj.IsInitialized())
        data_vector, pscores_vector, dscores_vector, ds_old_scores_vector, ds_start_scores_vector = self.communicator.get_scores_by_type(idxs, actions, gsmis, protein, receptor, dockobj)

        self.past_dockobjs.append(dockobj)
        self.past_receptors.append(receptor)
        orders = []
        for env_idx in range(len(data_vector)):
            if self.sort == 'dscores':
                order = np.argsort(dscores_vector[env_idx])
            elif self.sort == 'pscores':
                order = np.argsort(pscores_vector[env_idx])
            elif self.sort == 'iscores':
                order = np.argsort(ds_start_scores_vector[env_idx])
            elif self.sort == 'hscores':
                hscores = [np.quantile(np.clip(scoreset, None, 0), 0.) for scoreset in ds_old_scores_vector[env_idx]]
                order = np.argsort(hscores)
            else:
                assert (False)
            orders.append(order)

        sorted_data_vector = []
        for env_idx in range(len(data_vector)):
            data = [data_vector[env_idx][i] for i in orders[env_idx]]
            sorted_data_vector.append(data)

        return sorted_data_vector

    def choose_action(self, pdb_string):
        with self.logger("choose_action") as logger:
            with tempfile.TemporaryDirectory() as dirname:
                with open("{}/test.pdb".format(dirname), 'w') as f:
                    f.write(pdb_string)
                pdb = oechem.OEMol()
                prot = oechem.OEMol()
                lig = oechem.OEMol()
                wat = oechem.OEGraphMol()
                other = oechem.OEGraphMol()
                ifs = oechem.oemolistream()
                ifs.SetFlavor(oechem.OEFormat_PDB,
                              oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA | oechem.OEIFlavor_PDB_ALTLOC)  # noqa
                if not ifs.open("{}/test.pdb".format(dirname)):
                    logger.log("crap")
                oechem.OEReadMolecule(ifs, pdb)
                ifs.close()
                if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
                    logger.failure("could not split complex. exiting", exit_all=True)
                else:
                    logger.log("atom sizes for incoming step", len(list(lig.GetAtoms())), len(list(prot.GetAtoms())),
                               len(list(wat.GetAtoms())), len(list(other.GetAtoms())))
                original_smiles_vector, oeclean_smiles_vector = self.communicator.get_new_action_sets(aligner=lig)
                data_vector = self.getscores(original_smiles_vector,
                                      oeclean_smiles_vector,
                                      prot,
                                      lig,
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
