from xmlrpc.client import ServerProxy, Binary, Fault

import numpy as np
from gym import spaces
from itertools import combinations
from openeye import oechem, oeshape, oeomega, oemolprop, oemedchem
from rdkit import Chem

from rlmm.environment import molecules
from rlmm.utils.config import Config
from rlmm.utils.loggers import make_message_writer


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

class RocsMolAligner:
    def __init__(self, reference_mol=None):
        if reference_mol is not None:
            self.refmol = oechem.OEMol(reference_mol)
        else:
            self.refmol = None

    def update_reference_mol(self, oemol):
        self.refmol = oechem.OEMol(oemol)

    def get_reference_mol(self):
        return oechem.OEMol(self.refmol)

    def __call__(self, new_smile):
        fitfs = oechem.oemolistream()
        fitfs.SetFormat(oechem.OEFormat_SMI)
        fitfs.openstring(new_smile)

        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
        omegaOpts.SetStrictAtomTypes(False)
        omegaOpts.SetSampleHydrogens(True)
        omegaOpts.SetMaxSearchTime(30)
        omegaOpts.SetFixDeleteH(True)
        omega = oeomega.OEOmega(omegaOpts)

        options = oeshape.OEROCSOptions()
        overlayoptions = oeshape.OEOverlayOptions()
        overlayoptions.SetOverlapFunc(
            oeshape.OEOverlapFunc(oeshape.OEAnalyticShapeFunc(), oeshape.OEAnalyticColorFunc()))
        options.SetOverlayOptions(overlayoptions)
        options.SetNumBestHits(10)
        options.SetConfsPerHit(1)
        options.SetMaxHits(10000)
        rocs = oeshape.OEROCS(options)

        for fitmol in fitfs.GetOEMols():
            for enantiomer in oeomega.OEFlipper(fitmol.GetActive(), 5, True):
                enantiomer = oechem.OEMol(enantiomer)
                ret_code = omega.Build(enantiomer)
                if ret_code != oeomega.OEOmegaReturnCode_Success:
                    pass
                else:
                    rocs.AddMolecule(oechem.OEMol(enantiomer))

        for res in rocs.Overlay(self.refmol):
            outmol = oechem.OEMol(res.GetOverlayConf())
            good_mol = oechem.OEMol(outmol)
            oechem.OEAddExplicitHydrogens(good_mol)
            oechem.OEClearSDData(good_mol)
            oeshape.OEDeleteCompressedColorAtoms(good_mol)
            oeshape.OEClearCachedSelfColor(good_mol)
            oeshape.OEClearCachedSelfShape(good_mol)
            oeshape.OERemoveColorAtoms(good_mol)
            return good_mol

        return None


def filter_smiles(smis):
    '''
    :param smis: list of smiles
    :return: (oe graph mols, smiles)
    '''
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    smiles = "\n".join(list(smis))
    ims.openstring(smiles)

    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SMI)
    oms.openstring()

    filt = oemolprop.OEFilter(oemolprop.OEFilterType_BlockBuster)

    goods = []
    for i, mol in enumerate(ims.GetOEGraphMols()):
        if filt(mol):
            oechem.OEWriteMolecule(oms, mol)
            goods.append(i)
    actions = str(oms.GetString().decode("utf-8"))
    actions = actions.split("\n")

    oms.close()
    ims.close()
    return [smis[i] for i in goods], actions


class FastRocsActionSpace:
    class Config(Config):
        def __init__(self, configs):
            self.host = configs['host']
            self.space_size = configs['space_size']

        def get_obj(self):
            return FastRocsActionSpace(self)

    def __init__(self, config):
        self.config = config
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            pass

    def setup(self, starting_ligand_file):
        mol = oechem.OEMol()
        ifs = oechem.oemolistream(starting_ligand_file)
        oechem.OEReadMolecule(ifs, mol)
        self.set_mole_aligner(mol)

    def get_new_action_set(self, aligner=None):
        with self.logger("get_new_action_set") as logger:
            if aligner is not None:
                self.set_mole_aligner(aligner)
            mols = self.fastrocs_query(self.mol_aligner, self.config.space_size, self.config.host)
            smiles = [oechem.OEMolToSmiles(mol) for mol in mols]

        return mols, smiles

    def apply_action(self, mol, action=None):
        self.mol_aligner = oechem.OEMol(mol)

    def set_mole_aligner(self, oemol):
        self.mol_aligner = oechem.OEMol(oemol)

    def get_aligned_action(self, oemol: oechem.OEMolBase, oe_smiles: str):
        return oemol, oechem.OEMol(oemol), oe_smiles, oe_smiles

    def get_gym_space(self):
        # TODO
        return spaces.Discrete(2)

    def fastrocs_query(self, qmol, numHits, host):
        with self.logger("fastrocs_query") as logger:
            ofs = oechem.oemolostream()
            ofs.SetFormat(oechem.OEFormat_OEB)
            ofs.openstring()
            oechem.OEWriteMolecule(ofs, qmol)
            bytes = ofs.GetString()

            s = ServerProxy("http://" + host)
            data = Binary(bytes)
            idx = s.SubmitQuery(data, numHits)

            dargs = {'altStarts' :  None,  'tversky' : True, 'shapeOnly' : False}

            first = False
            while True:
                try:
                    current, total = s.QueryStatus(idx, numHits, 'oeb', 'oeb', dargs)
                except Fault as e:
                    logger.error((str(e)))
                    return 1

                if total == 0:
                    continue

                if first:
                    logger.log("%s/%s" % ("current", "total"))
                    first = False
                logger.log("%i/%i" % (current, total))
                if total <= current:
                    break
            results = s.QueryResults(idx)
            ifs = oechem.oemolistream()
            ifs.openstring(results.data)
            ifs.SetFormat(oechem.OEFormat_OEB)
            mols = []
            for mol in ifs.GetOEMols():
                good_mol = oechem.OEMol(mol)
                oechem.OEAddExplicitHydrogens(good_mol)
                oechem.OEClearSDData(good_mol)
                oeshape.OEDeleteCompressedColorAtoms(good_mol)
                oeshape.OEClearCachedSelfColor(good_mol)
                oeshape.OEClearCachedSelfShape(good_mol)
                oeshape.OERemoveColorAtoms(good_mol)
                mols.append(good_mol)
        return mols


class MoleculePiecewiseGrow:
    class Config(Config):
        def __init__(self, configs):
            config_default = {
                'atoms': ['C', 'O', "N", 'F', 'S', 'H', 'Br', 'Cl'],
                'allow_removal': True,
                'allowed_ring_sizes': [3, 4, 5, 6, 7, 8],
                'allow_no_modification': True,
                'allow_bonds_between_rings': False
            }
            config_default.update(configs)
            self.atoms = set(config_default['atoms'])
            self.allow_removal = config_default['allow_removal']
            self.allowed_ring_sizes = config_default['allowed_ring_sizes']
            self.allow_no_modification = config_default['allow_no_modification']
            self.allow_bonds_between_rings = config_default['allow_bonds_between_rings']

        def get_obj(self):
            return MoleculePiecewiseGrow(self)

    def __init__(self, config):
        self.config = config
        self.aligner = RocsMolAligner()

    def setup(self, starting_ligand_file):
        self.start_smiles = Chem.MolFromMol2File(starting_ligand_file)
        if np.abs(Chem.GetFormalCharge(self.start_smiles) - int(Chem.GetFormalCharge(self.start_smiles))) != 0:
            print("NONINTEGRAL START CHARGE", Chem.GetFormalCharge(self.start_smiles))
        Chem.SanitizeMol(self.start_smiles)
        Chem.AssignStereochemistry(self.start_smiles, cleanIt=True, force=True)

        mol = oechem.OEMol()
        ifs = oechem.oemolistream(starting_ligand_file)
        oechem.OEReadMolecule(ifs, mol)
        self.mol_aligner = mol
        ifs.close()
        self.mol = molecules.Molecule(self.config.atoms, self.start_smiles,
                                      allow_removal=self.config.allow_removal,
                                      allow_no_modification=self.config.allow_no_modification,
                                      allow_bonds_between_rings=self.config.allow_no_modification,
                                      allowed_ring_sizes=self.config.allowed_ring_sizes, max_steps=100)
        self.mol.initialize()

    def get_new_action_set(self, aligner=None):
        if aligner is not None:
            self.set_mole_aligner(aligner)
        actions = list(self.mol.get_valid_actions())
        original_smiles, oeclean_smiles = filter_smiles(actions)
        return original_smiles, oeclean_smiles

    def apply_action(self, mol, action):
        _ = self.mol.step(action)
        self.mol_aligner = oechem.OEMol(mol)
        self.aligner.update_reference_mol(mol)

    def set_mole_aligner(self, oemol):
        self.mol_aligner = oechem.OEMol(oemol)
        self.aligner.update_reference_mol(oemol)

    def get_aligned_action(self, original_smiles, oe_smiles):
        new_mol = self.aligner(oe_smiles)
        return new_mol, oechem.OEMol(new_mol), oe_smiles, original_smiles

    def get_gym_space(self):
        # TODO
        return spaces.Discrete(2)


class EuclidanActionSpace:
    class Config(Config):
        def __init__(self, configs):
            self.ligand_only = configs['ligand_only']
            self.minimize = configs['minimize']

        def get_obj(self):
            return EuclidanActionSpace(self)

    def __init__(self, config: Config):
        self.config = config

    def apply_action_simulation(self, action, simulation):
        simulation.translate(*action, ligand_only=self.config.ligand_only, minimize=self.config.minimize)

    def get_gym_space(self):
        return spaces.Discrete(2)
