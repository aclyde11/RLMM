import tempfile
import pickle
import numpy as np
from openeye import oechem, oedocking
from simtk import unit
from sklearn.decomposition import PCA
from rlmm.utils.loggers import make_message_writer


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


    def get_scores_by_type(self, idxs_vector, actions_vector, gsmis_vector, protein, receptor, dockobj):
        """ this is a serial version, has to be parallelized with mpi """
        pscores_vector = []
        dscores_vector = []
        ds_old_scores_vector = []
        ds_start_scores_vector = []

        data_vector = []
        for env_idx in range(len(idxs_vector)):
            pscores = []
            dscores = []
            ds_old_scores = []
            ds_start_scores = []

            data = []

            idxs = idxs_vector[env_idx]
            actions = actions_vector[env_idx]
            gsmis = gsmis_vector[env_idx]
            for idx in idxs:
                try:
                    res = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                    if res is None:
                        continue
                    new_mol, new_mol2, gs, action = res
                    dockedpose = oechem.OEMol()
                    dockobj.DockMultiConformerMolecule(dockedpose, new_mol, 1)
                    ds = dockedpose.GetEnergy()
                    ps = dockobj.ScoreLigand(new_mol)

                    ds_old = []
                    ds_start = None
                    if self.start_dobj is not None:
                        dockedpose2 = oechem.OEMol()
                        newmol2 = oechem.OEMol(new_mol)
                        self.start_dobj.DockMultiConformerMolecule(dockedpose2, newmol2, 1)
                        ds_start = dockedpose2.GetEnergy()
                    if self.track_hscores:
                        for olddobj in self.past_dockobjs:
                            dockedpose2 = oechem.OEMol()
                            newmol2 = oechem.OEMol(new_mol)
                            olddobj.DockMultiConformerMolecule(dockedpose2, newmol2, 1)
                            ds_old.append(dockedpose2.GetEnergy())

                    new_mol2 = oechem.OEMol(new_mol)
                    oechem.OEAssignAromaticFlags(new_mol)
                    oechem.OEAssignAromaticFlags(new_mol2)
                    oechem.OEAddExplicitHydrogens(new_mol)
                    oechem.OEAddExplicitHydrogens(new_mol2)
                    oechem.OE3DToInternalStereo(new_mol)
                    oechem.OE3DToInternalStereo(new_mol2)
                    gs = oechem.OECreateSmiString(
                        new_mol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens
                                 | oechem.OESMILESFlag_Isotopes | oechem.OESMILESFlag_BondStereo
                                 | oechem.OESMILESFlag_AtomStereo)

                    if self.track_hscores:
                        ds_old_scores.append(ds_old)
                    ds_start_scores.append(ds_start)
                    dscores.append(ds)
                    pscores.append(ps)

                    data.append((new_mol, new_mol2, gs, action))
                except Exception as p:
                    logger.error(p)
                    traceback.print_tb(p.__traceback__)

                    continue
            # this has to be sent to each env
            self.env.data['dscores'].append(dscores)
            self.env.data['pscores'].append(pscores)
            self.env.data['iscores'].append(ds_start_scores)
            self.env.data['hscores'].append(ds_old_scores)
            #################################
            data_vector.append(data)
            pscores_vector.append(pscores)
            dscores_vector.append(dscores)
            ds_old_scores_vector.append(ds_old_scores)
            ds_start_scores_vector.append(ds_start_scores)

        return data_vector, pscores_vector, dscores_vector, ds_old_scores_vector. ds_start_scores_vector

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
