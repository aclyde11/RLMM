import tempfile
import traceback

import numpy as np
from openeye import oechem, oedocking
from sklearn.decomposition import PCA

from rlmm.utils.loggers import make_message_writer

class RandomPolicy:

    def __init__(self, env, sort='dscores', return_docked_pose=False, num_returns=-1, orig_pdb=None, useHybrid=False,
                 trackHScores=True):
        self.logger = make_message_writer(env.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.sort = sort
            self.return_docked_pose = return_docked_pose
            self.num_returns = num_returns
            self.env = env


    def getscores(self, actions, gsmis, prot, lig, num_returns=10, return_docked_pose=False):
        with self.logger("getscores") as logger:
            if num_returns <= 0:
                num_returns = len(actions) - 1
            logger.log("Action space is ", len(actions))
            idxs = list(np.random.choice(len(actions), min(num_returns, len(actions) - 1), replace=False).flatten())





            data = []
            for idx in idxs:
                try:
                    res = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                    if res is None:
                        logger.error("Alignment failed and returned none for ", gsmis[idx])
                        continue
                    new_mol, new_mol2, gs, action = res

                    data.append((new_mol, new_mol2, gs, action))
                except Exception as p:
                    logger.error(p)
                    traceback.print_tb(p.__traceback__)

                    continue



        return data

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
                original_smiles, oeclean_smiles = self.env.action.get_new_action_set(aligner=lig)
                data = self.getscores(original_smiles, oeclean_smiles, prot, lig, num_returns=self.num_returns,
                                      return_docked_pose=self.return_docked_pose)
                not_worked = True
                idxs = list(range(len(data)))
                idx = idxs.pop(0)
                while not_worked:
                    try:
                        new_mol, new_mol2, gs, action = data[idx]

                        self.env.systemloader.reload_system(gs, new_mol, "{}/test.pdb".format(dirname))
                        self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                           self.env.openmm_simulation)
                        not_worked = False
                    except Exception as e:
                        logger.error("Could not buid system for smiles", gs, "with exception", e)
                        traceback.print_tb(e.__traceback__)

                        if len(idxs) == 0:
                            logger.failure("No system could build", exit_all=True)
                        idx = idxs.pop(0)
                self.env.action.apply_action(new_mol2, action)

        return new_mol2, action

class ExpertPolicy:

    def __init__(self, env, sort='dscores', return_docked_pose=False, num_returns=-1, orig_pdb=None, useHybrid=False,
                 trackHScores=True):
        self.logger = make_message_writer(env.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.sort = sort
            self.return_docked_pose = return_docked_pose
            self.num_returns = num_returns
            self.env = env

            self.orig_pdb = orig_pdb
            self.start_dobj = None
            self.start_receptor = None
            self.track_hscores = trackHScores
            assert (not (not self.track_hscores and self.sort == 'hscores'))
            self.past_receptors = []
            self.past_dockobjs = []
            self.past_coordinates = []
            self.pca = PCA(2)
            self.dockmethod = oedocking.OEDockMethod_Hybrid if useHybrid else oedocking.OEDockMethod_Chemgauss4
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
                    logger.failure("Could not split complex", exit_all=True)

                self.start_receptor = oechem.OEGraphMol()
                logger.log("Building initial receptor file...")
                oedocking.OEMakeReceptor(self.start_receptor, prot, lig)

                self.start_dobj = oedocking.OEDock(self.dockmethod)
                self.start_dobj.Initialize(self.start_receptor)
                assert (self.start_dobj.IsInitialized())
                logger.log("done")

    def getscores(self, actions, gsmis, prot, lig, num_returns=10, return_docked_pose=False):
        with self.logger("getscores") as logger:
            if num_returns <= 0:
                num_returns = len(actions) - 1
            logger.log("Action space is ", len(actions))
            idxs = list(np.random.choice(len(actions), min(num_returns, len(actions) - 1), replace=False).flatten())

            protein = oechem.OEMol(prot)
            receptor = oechem.OEGraphMol()
            logger.log("Creating receptor from recent pdb, this might take awhile")
            oedocking.OEMakeReceptor(receptor, protein, lig)
            dockobj = oedocking.OEDock(self.dockmethod)
            dockobj.Initialize(receptor)
            assert (dockobj.IsInitialized())
            logger.log("done")

            pscores = []
            dscores = []
            ds_old_scores = []
            ds_start_scores = []

            data = []
            for idx in idxs:
                try:
                    res = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                    if res is None:
                        logger.error("Alignment failed and returned none for ", gsmis[idx])
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
                    logger.log(
                        "Proposed action data... Pose Score {}, Dock Score {}, Init Score {}, History Scores {}".format(
                            ps, ds, ds_start, ds_old))
                    # if return_docked_pose:
                    #     new_mol = oechem.OEMol(dockedpose)
                    #     new_mol2 = oechem.OEMol(dockedpose)

                    data.append((new_mol, new_mol2, gs, action))
                except Exception as p:
                    logger.error(p)
                    traceback.print_tb(p.__traceback__)

                    continue

            self.past_dockobjs.append(dockobj)
            self.past_receptors.append(receptor)
            logger.log("Sorting on", self.sort)
            if self.sort == 'dscores':
                order = np.argsort(dscores)
                logger.log([dscores[i] for i in order])
            elif self.sort == 'pscores':
                order = np.argsort(pscores)
                logger.log([pscores[i] for i in order])
            elif self.sort == 'iscores':
                order = np.argsort(ds_start_scores)
                logger.log([ds_start_scores[i] for i in order])
            elif self.sort == 'hscores':
                hscores = [np.quantile(np.clip(scoreset, None, 0), 0.) for scoreset in ds_old_scores]
                order = np.argsort(hscores)
                logger.log([hscores[i] for i in order])
            else:
                assert (False)

            self.env.data['dscores'].append(dscores)
            self.env.data['pscores'].append(pscores)
            self.env.data['iscores'].append(ds_start_scores)
            self.env.data['hscores'].append(ds_old_scores)
            data = [data[i] for i in order]
        return data

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
                original_smiles, oeclean_smiles = self.env.action.get_new_action_set(aligner=lig)
                data = self.getscores(original_smiles, oeclean_smiles, prot, lig, num_returns=self.num_returns,
                                      return_docked_pose=self.return_docked_pose)
                not_worked = True
                idxs = list(range(len(data)))
                idx = idxs.pop(0)
                while not_worked:
                    try:
                        new_mol, new_mol2, gs, action = data[idx]

                        self.env.systemloader.reload_system(gs, new_mol, "{}/test.pdb".format(dirname))
                        self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                           self.env.openmm_simulation)
                        not_worked = False
                    except Exception as e:
                        logger.error("Could not buid system for smiles", gs, "with exception", e)
                        traceback.print_tb(e.__traceback__)

                        if len(idxs) == 0:
                            logger.failure("No system could build", exit_all=True)
                        idx = idxs.pop(0)
                self.env.action.apply_action(new_mol2, action)

        return new_mol2, action
