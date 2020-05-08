import random

from gym import spaces
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolAlign

from rlmm.environment import molecules
from rlmm.utils.config import Config


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

def get_and_align(new_smile, reference_mol):
    from openeye import oechem, oeshape, oeomega
    import numpy as np
    fitfs = oechem.oemolistream()
    fitfs.SetFormat(oechem.OEFormat_SMI)
    fitfs.openstring(new_smile)

    refmol = oechem.OEMol(reference_mol)
    print("Ref. Title:", refmol.GetTitle(), "Num Confs:", refmol.NumConfs())


    omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
    omegaOpts.SetStrictAtomTypes(False)
#    omegaOpts.SetStrictFrags(False)
#    omegaOpts.SetStrictAtomTypes(False)
#    omegaOpts.SetFromCT(True)
#    omegaOpts.SetMaxConfs(400)

    omega = oeomega.OEOmega(omegaOpts)


    options = oeshape.OEROCSOptions()
    overlayoptions = oeshape.OEOverlayOptions()
    overlayoptions.SetOverlapFunc(oeshape.OEOverlapFunc(oeshape.OEAnalyticShapeFunc(), oeshape.OEAnalyticColorFunc()))
    options.SetOverlayOptions(overlayoptions)
    options.SetNumBestHits(10)
#    options.ClearCutoffs()
    options.SetConfsPerHit(1)
    options.SetMaxHits(10000)
    rocs = oeshape.OEROCS(options)

    for fitmol in fitfs.GetOEMols():
        for enantiomer in oeomega.OEFlipper(fitmol.GetActive(), 3, False):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code != oeomega.OEOmegaReturnCode_Success:
                pass
            else:
                rocs.AddMolecule(oechem.OEMol(enantiomer))

    for res in rocs.Overlay(refmol):
        outmol = oechem.OEMol(res.GetOverlayConf())
        good_mol = oechem.OEMol(outmol)
        # good_mol =  mols[np.argmax(scores)]
        oechem.OEAddExplicitHydrogens(good_mol)
        oechem.OEClearSDData(good_mol)
        oeshape.OEDeleteCompressedColorAtoms(good_mol)
        oeshape.OEClearCachedSelfColor(good_mol)
        oeshape.OEClearCachedSelfShape(good_mol)
        oeshape.OERemoveColorAtoms(good_mol)
        return good_mol

    return None


def filter_smiles(smis):
    from openeye import oechem, oemolprop
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    smiles = "\n".join(list(smis))
    ims.openstring(smiles)

    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SMI)
    oms.openstring()

    filt = oemolprop.OEFilter(oemolprop.OEFilterType_BlockBuster)

    goods =[]
    for i,mol in enumerate(ims.GetOEGraphMols()):
        if filt(mol):
            oechem.OEWriteMolecule(oms, mol)
            goods.append(i)
    actions = str(oms.GetString().decode("utf-8") )
    actions = actions.split("\n")

    oms.close()
    ims.close()
    return [smis[i] for i in goods], actions

class LigandTransformSpace:
    class Config(Config):
        def __init__(self, configs):
            self.ligand_only = configs['ligand_only']
            self.minimize = configs['minimize']

        def get_obj(self):
            return LigandTransformSpace(self)

    def __init__(self, config):
        self.config = config

    def setup(self, ligand):
        from openeye import oechem
        self.start_smiles = Chem.MolFromMol2File(ligand)
        print("START CHARGE", Chem.GetFormalCharge(self.start_smiles))
        Chem.SanitizeMol(self.start_smiles)
        Chem.AssignStereochemistry(self.start_smiles, cleanIt=True, force=True)

        mol = oechem.OEMol()
        ifs = oechem.oemolistream(ligand)
        oechem.OEReadMolecule(ifs, mol)
        self.mol_aligner = mol
        ifs.close()
        self.mol = molecules.Molecule({'C', 'O', "N", 'F', 'S'}, self.start_smiles, allow_removal=False,
                                      allow_no_modification=False,
                                      allow_bonds_between_rings=True,
                                      allowed_ring_sizes=[5,6],max_steps = 100)
        print("NEWSMILE", self.start_smiles)
        self.mol.initialize()

    def get_new_action_set(self):
        actions = list(self.mol.get_valid_actions())
        actions, gsmis = filter_smiles(actions)
        return actions, gsmis

    def apply_action(self, mol, action):
        from openeye import oechem

        res = self.mol.step(action)
        print("NEWSMILE", action)

        self.mol_aligner = oechem.OEMol(mol)

    def get_aligned_action(self, actions, gsmis):
        from openeye import oechem

        gs = gsmis
        action = actions
        new_mol = get_and_align(gs, self.mol_aligner)



        return new_mol, oechem.OEMol(new_mol), gs, action

    def get_gym_space(self):
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
