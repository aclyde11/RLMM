"""
Unit and regression test for the rlmm package.
"""

# import rlmm.environment.openmmEnv
# Import package, test suite, and other packages as needed
# import rlmm.environment.openmmWrapper
# import rlmm.environment.systemloader

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdmolops, AllChem
import random
from rlmm.environment import molecules
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

def test_PDBLoader_get_mobile():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    assert(644 == env.systemloader.get_mobile())

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
        s=True
    except ValueError:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        s=False

    if mol.GetNumConformers() <= 0:
        mol=None
        s=False

    return mol,s

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

def test_load_test_system():
    config = Config.load_yaml('rlmm/tests/test_config.yaml')
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))

    for i in range(15):
        env.step([0.05, 0.05, 0.05])
        env.openmm_simulation.get_pdb(file_name='rlmm/tests/test_out1.pdb')




if __name__ == '__main__':
    test_load_test_system()
