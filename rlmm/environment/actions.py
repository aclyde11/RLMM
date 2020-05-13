import numpy as np
from gym import spaces
from openeye import oechem, oeshape, oeomega, oemolprop
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolAlign
import subprocess
from rlmm.environment import molecules
from rlmm.utils.config import Config
import tempfile
from rlmm.rl.fastrocs import fastrocs_query

def getMCS(m1, m2):
    mcs = rdFMCS.FindMCS([m1, m2], ringMatchesRingOnly=True)
    core = Chem.MolFromSmarts(mcs.smartsString)
    match = m2.GetSubstructMatches(core)[0]
    core_m = Chem.EditableMol(m2)
    for idx in range(m2.GetNumAtoms() - 1, -1, -1):
        if idx not in match:
            core_m.RemoveAtom(idx)
    core_m = core_m.GetMol()
    try:
        Chem.SanitizeMol(core_m)
    except ValueError:
        pass

    return core, core_m


def get_3d(mol, ref):
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    try:
        AllChem.ConstrainedEmbed(mol, ref)
        s = True
    except ValueError:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        s = False

    if mol.GetNumConformers() <= 0:
        mol = None
        s = False

    return mol, s


def test_new(m_new, mol_aligner):
    _, coreM = getMCS(m_new, mol_aligner)
    m_new, s = get_3d(m_new, coreM)
    try:
        assert (m_new is not None)
        if not s:
            match1 = m_new.GetSubstructMatch(coreM)
            match2 = mol_aligner.GetSubstructMatch(coreM)
            rdMolAlign.AlignMol(m_new, mol_aligner, atomMap=list(zip(match1, match2)))
    except RuntimeError:
        # print("step {} Error on fallback alignment")
        return None
    except AssertionError:
        # print("step {} error on total confgen")
        return None
    return m_new


def run_check(data):
    res, mol_aligner = data
    m_new = Chem.MolFromSmiles(res)
    m_new = test_new(m_new, mol_aligner)
    return m_new, res

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
        overlayoptions.SetOverlapFunc(oeshape.OEOverlapFunc(oeshape.OEAnalyticShapeFunc(), oeshape.OEAnalyticColorFunc()))
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
            self.host = set(configs['host'])
            self.space_size = configs['space_size']

        def get_obj(self):
            return FastRocsActionSpace(self)

    def __init__(self, config):
        self.config = config

    def setup(self, starting_ligand_file):
        mol = oechem.OEMol()
        ifs = oechem.oemolistream(starting_ligand_file)
        oechem.OEReadMolecule(ifs, mol)
        self.set_mole_aligner(mol)

    def get_new_action_set(self, aligner=None):
        if aligner is not None:
            self.set_mole_aligner(aligner)
        mols = fastrocs_query(self.mol_aligner, self.config.space_size, self.config.host)
        smiles = [oechem.OEMolToSmiles(mol) for mol in mols]

        return mols, smiles

    def apply_action(self, mol, action=None):
        self.mol_aligner = oechem.OEMol(mol)

    def set_mole_aligner(self, oemol):
        self.mol_aligner = oechem.OEMol(oemol)

    def get_aligned_action(self, oemol : oechem.OEMolBase,  oe_smiles : str):
        return oemol, oechem.OEMol(oemol), oe_smiles, oe_smiles

    def get_gym_space(self):
        #TODO
        return spaces.Discrete(2)


class MoleculePiecewiseGrow:
    class Config(Config):
        def __init__(self, configs):
            config_default = {
                'atoms' : ['C', 'O', "N", 'F', 'S', 'H', 'Br', 'Cl'],
                'allow_removal' : True,
                'allowed_ring_sizes' : [3, 4, 5, 6, 7, 8],
                'allow_no_modification' : True,
                'allow_bonds_between_rings' : False
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
        #TODO
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
