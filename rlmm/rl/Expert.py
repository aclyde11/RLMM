import subprocess
from itertools import combinations
import numpy as np
from openeye import oechem, oedocking, oemedchem
from simtk import unit
import tempfile
from rlmm.utils.loggers import  make_message_writer

class FastRocsPolicy:

    def __init__(self, env, header='python /vol/ml/aclyde/fastrocs/ShapeDatabaseClient.py dgx1:8080', hits=100,
                 step_size=3.5, orig_pdb=None):
        self.env = env
        self.header = header
        self.hits = hits
        self.step_size = step_size
        self.orig_pdb = orig_pdb
        self.start_receptor = None
        self.start_dobj = None
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
                print("crap")
            self.start_receptor = oechem.OEGraphMol()
            oedocking.OEMakeReceptor(self.start_receptor, prot, lig)
            self.start_dobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
            self.start_dobj.Initialize(self.start_receptor)
        self.store = [(self.start_receptor, self.start_dobj)]

    def getscores(self, actions, gsmis, prot, ligand, num_returns=10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions) - 1
        print("Action space is ", len(actions))
        idxs = list(range(min(num_returns, len(actions) - 1)))

        protein = oechem.OEMol(prot)
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, ligand)
        dockobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
        dockobj.Initialize(receptor)
        scores = []
        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = actions[idx], actions[idx], gsmis[idx], gsmis[idx]
                dockedpose = oechem.OEMol()
                dockobj.DockMultiConformerMolecule(dockedpose, new_mol, 1)
                ds = dockedpose.GetEnergy()
                ps = dockobj.ScoreLigand(new_mol)
                ds2 = [ds]
                for _, dobj in self.store:
                    dockedpose2 = oechem.OEMol()
                    newmol2 = oechem.OEMol(new_mol)
                    dobj.DockMultiConformerMolecule(dockedpose2, newmol2, 1)
                    ds2.append(dockedpose2.GetEnergy())

                ds2 = np.clip(ds2 + [ps], None, 0)
                ds2avg = np.mean(ds2)
                print("SCORE", ds2avg, ds, ps, ds2)

                if return_docked_pose:
                    new_mol = oechem.OEMol(dockedpose)
                    new_mol2 = oechem.OEMol(dockedpose)
                oechem.OEAddExplicitHydrogens(new_mol2)
                oechem.OEAddExplicitHydrogens(new_mol)
                data.append((new_mol, new_mol2, gs, action))
                scores.append(ds2avg)
            except Exception as e:
                print(e)
                continue
        self.store.append((receptor, dockobj))
        order = np.argsort(scores)
        data = [data[i] for i in order]
        return data

    def choose_action(self):
        self.env.openmm_simulation.get_pdb("test.pdb")
        pdb = oechem.OEMol()
        prot = oechem.OEMol()
        lig = oechem.OEMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        ifs = oechem.oemolistream("test.pdb")
        oechem.OEReadMolecule(ifs, pdb)
        ifs.close()
        if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
            print("crap")
            exit()

        oemols, smiles = self.env.action.get_new_action_set(aligner=lig)

        data = self.getscores(oemols, smiles, prot, lig, num_returns=-1, return_docked_pose=False)
        not_worked = True
        idxs = list(range(len(data)))
        idx = idxs.pop(0)
        counter = 0
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = data[idx]
                self.env.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                   ln=self.env.systemloader,
                                                                                   stepSize=self.step_size * unit.femtoseconds)

                not_worked = False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)
        return new_mol2, action


class RandomPolicy:

    def __init__(self, env, return_docked_pose=False, num_returns=-1, step_size=3.5):
        self.return_docked_pose = return_docked_pose
        self.num_returns = num_returns
        self.env = env
        self.step_size = step_size

    def getscores(self, actions, gsmis, prot, num_returns=10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions) - 1
        print("Action space is ", len(actions))
        idxs = list(np.random.choice(len(actions), min(num_returns, len(actions) - 1), replace=False).flatten())

        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                data.append((new_mol, new_mol2, gs, action))
            except:
                continue
        return data

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

            actions, gsmis = self.env.action.get_new_action_set(aligner=lig)
            data = self.getscores(actions, gsmis, prot, num_returns=self.num_returns,
                                  return_docked_pose=self.return_docked_pose)
            not_worked = True
            idxs = list(range(len(data)))
            idx = idxs.pop(0)
            counter = 0
            while not_worked:
                try:
                    new_mol, new_mol2, gs, action = data[idx]
                    self.env.systemloader.reload_system(gs, new_mol, "{}/test.pdb".format(dirname))
                    self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                       ln=self.env.systemloader,
                                                                                       stepSize=self.step_size * unit.femtoseconds,
                                                                                       prior_sim=self.env.openmm_simulation.simulation)
                    not_worked = False
                except Exception as e:
                    print(e)
                    if len(idxs) == 0:
                        print("mega fail")
                        exit()
                    idx = idxs.pop(0)
            self.env.action.apply_action(new_mol2, action)

        return new_mol2, action

def get_mols_from_frags(this_smiles, old_smiles=None):
    if old_smiles is None:
        old_smiles = []
    fragfunc = GetFragmentationFunction()
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, this_smiles)
    frags = [f for f in fragfunc(mol)]
    len_frags = len(frags)

    for smile in old_smiles:
        mol2 = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol2, smile)
        frags += [f for f in fragfunc(mol2)]

    oechem.OEThrow.Info("%d number of fragments generated" % len(frags))

    fragcombs = GetFragmentCombinations(mol, frags, frag_number=len_frags)
    oechem.OEThrow.Info("%d number of fragment combinations generated" % len(fragcombs))

    smiles = set()
    for frag in fragcombs:
        if oechem.OEDetermineComponents(frag)[0] == 1:
            smiles = smiles.union(oechem.OEMolToSmiles(frag))
    return smiles

def IsAdjacentAtomBondSets(fragA, fragB):
    for atomA in fragA.GetAtoms():
        for atomB in fragB.GetAtoms():
            if atomA.GetBond(atomB) is not None:
                return True
    return False

def IsAdjacentAtomBondSetCombination(fraglist):
    parts = [0] * len(fraglist)
    nrparts = 0
    for idx, frag in enumerate(fraglist):
        if parts[idx] != 0:
            continue
        nrparts += 1
        parts[idx] = nrparts
        TraverseFragments(frag, fraglist, parts, nrparts)
    return (nrparts == 1)

def TraverseFragments(actfrag, fraglist, parts, nrparts):
    for idx, frag in enumerate(fraglist):
        if parts[idx] != 0:
            continue
        if not IsAdjacentAtomBondSets(actfrag, frag):
            continue
        parts[idx] = nrparts
        TraverseFragments(frag, fraglist, parts, nrparts)

def CombineAndConnectAtomBondSets(fraglist):
    combined = oechem.OEAtomBondSet()
    for frag in fraglist:
        for atom in frag.GetAtoms():
            combined.AddAtom(atom)
        for bond in frag.GetBonds():
            combined.AddBond(bond)
    for atomA in combined.GetAtoms():
        for atomB in combined.GetAtoms():
            if atomA.GetIdx() < atomB.GetIdx():
                continue
            bond = atomA.GetBond(atomB)
            if bond is None:
                continue
            if combined.HasBond(bond):
                continue
            combined.AddBond(bond)
    return combined

def GetFragmentationFunction():
    return oemedchem.OEGetFuncGroupFragments

def GetFragmentAtomBondSetCombinations(mol, fraglist, desired_len):
    fragcombs = []
    nrfrags = len(fraglist)
    for n in range(max(desired_len - 3, 0), min(desired_len + 3, nrfrags)):
        for fragcomb in combinations(fraglist, n):
            frag = CombineAndConnectAtomBondSets(fragcomb)
            fragcombs.append(frag)
    return fragcombs


def GetFragmentCombinations(mol, fraglist, frag_number):
    fragments = []
    fragcombs = GetFragmentAtomBondSetCombinations(mol, fraglist, frag_number)
    for f in fragcombs:
        fragatompred = oechem.OEIsAtomMember(f.GetAtoms())
        fragbondpred = oechem.OEIsBondMember(f.GetBonds())
        fragment = oechem.OEGraphMol()
        adjustHCount = True
        oechem.OESubsetMol(fragment, mol, fragatompred, fragbondpred, adjustHCount)
        fragments.append(fragment)
    return fragments



class ExpertPolicy:

    def __init__(self, env, sort = 'dscores', return_docked_pose=False, num_returns=-1, orig_pdb=None):
        self.logger = make_message_writer(env.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.sort = sort
            self.return_docked_pose = return_docked_pose
            self.num_returns = num_returns
            self.env = env

            self.orig_pdb = orig_pdb
            self.start_dobj = None
            self.start_receptor = None

            self.past_receptors = []
            self.past_dockobjs = []
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
                    print("crap")
                    exit()

                self.start_receptor = oechem.OEGraphMol()
                logger.log("Building initial receptor file...")
                oedocking.OEMakeReceptor(self.start_receptor, prot, lig)
                self.start_dobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
                self.start_dobj.Initialize(self.start_receptor)
                assert(self.start_dobj.IsInitialized())
                logger.log("done")

    def getscores(self, actions, gsmis, prot, lig, num_returns=10, return_docked_pose=False):
        with self.logger("getscores") as logger:
            if num_returns <= 0:
                num_returns = len(actions) - 1
            logger.log("Action space is ", len(actions))
            idxs = list(np.random.choice(len(actions), min(num_returns, len(actions) - 1), replace=False).flatten())

            protein = oechem.OEMol(prot)
            receptor = oechem.OEGraphMol()
            oedocking.OEMakeReceptor(receptor, protein, lig)
            logger.log("Creating receptor from recent pdb, this might take awhile")
            dockobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
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
                    new_mol, new_mol2, gs, action = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
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
                        print(ds_start, "checking this out?")
                    for olddobj in self.past_dockobjs:
                        dockedpose2 = oechem.OEMol()
                        newmol2 = oechem.OEMol(new_mol)
                        olddobj.DockMultiConformerMolecule(dockedpose2, newmol2, 1)
                        ds_old.append(dockedpose2.GetEnergy())


                    ds_old_scores.append(ds_old)
                    ds_start_scores.append(ds_start)
                    dscores.append(ds)
                    pscores.append(ps)
                    logger.log("Proposed action data... Pose Score {}, Dock Score {}, Init Score {}, History Scores {}".format(ps, ds, ds_start, ds_old))
                    if return_docked_pose:
                        new_mol = oechem.OEMol(dockedpose)
                        new_mol2 = oechem.OEMol(dockedpose)

                    data.append((new_mol, new_mol2, gs, action))
                except:
                    continue
            hscores = [np.mean(np.clip(scoreset, None, 0)) for scoreset in ds_old_scores]
            self.past_dockobjs.append(dockobj)
            self.past_receptors.append(receptor)
            logger.log("Sorting on", self.sort)
            if self.sort == 'dscores':
                order = np.argsort(dscores)
            elif self.sort == 'pscores':
                order = np.argsort(pscores)
            elif self.sort == 'iscores':
                order = np.argsort(ds_start_scores)
            elif self.sort == 'hscores':
                order = np.argsort(hscores)
            else:
                assert(False)

            self.env.data['dscores'].append(dscores)
            self.env.data['pscores'].append(pscores)
            self.env.data['iscores'].append(ds_start_scores)
            self.env.data['hscores'].append(hscores)
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
                ifs = oechem.oemolistream("{}/test.pdb".format(dirname))
                oechem.OEReadMolecule(ifs, pdb)
                ifs.close()
                if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
                    logger.failure("could not split complex. exiting", exit_all=True)

                original_smiles, oeclean_smiles = self.env.action.get_new_action_set(aligner=lig)
                data = self.getscores(original_smiles, oeclean_smiles, prot, lig, num_returns=self.num_returns, return_docked_pose=self.return_docked_pose)
                not_worked = True
                idxs = list(range(len(data)))
                idx = idxs.pop(0)
                counter = 0
                while not_worked:
                    try:
                        new_mol, new_mol2, gs, action = data[idx]
                        self.env.systemloader.reload_system(gs, new_mol, "{}/test.pdb".format(dirname))
                        self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader,
                                                                                           ln=self.env.systemloader)
                        not_worked = False
                    except Exception as e:
                        out = oechem.oemolostream()
                        out.SetFormat(oechem.OEFormat_SDF)
                        out.openstring()
                        oechem.OEWriteMolecule(out, new_mol2)
                        print(out.GetString())

                        logger.log("Could not buid system for smiles", gs, "with exception", e)
                        if len(idxs) == 0:
                            logger.failure("No system could build", exit_all=True)
                        idx = idxs.pop(0)
                self.env.action.apply_action(new_mol2, action)

        return new_mol2, action
