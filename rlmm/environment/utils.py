import os
import os.path

import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from rdkit.Chem import AllChem


class Voxelizer:

    def __init__(self, pdb_structure, cache=None, write_cache=False, box_size=None, centers=None, voxelsize=None, validity_check=False, method='C', channel_first=False):
        self.boxsize = box_size
        self.centers = centers
        self.voxelsize = voxelsize
        self.method = method
        self.cache = cache
        self.validate = validity_check
        self.pdb_structure = pdb_structure
        self.channel_first = channel_first

        if cache is not None and os.path.isfile(cache):
            self.prot_vox_t = np.load(cache)
        else:
            prot = Molecule(pdb_structure)

            prot = prepareProteinForAtomtyping(prot, verbose=False)
            prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, buffer=0, voxelsize=self.voxelsize,
                                                                 boxsize=self.boxsize,
                                                                 center=self.centers, validitychecks=self.validate)

            self.prot_vox_t = self.reshape(prot_vox, prot_N)

            if write_cache and cache is not None:
                np.save(cache, self.prot_vox_t)

    def reshape(self, x, N):
        nchannels = x.shape[1]

        x = x.tranpose()
        if self.channel_first:
            return x.transpose().reshape([1, nchannels, N[0], N[1], N[2]])
        else:
            return x.transpose().reshape([1, N[0], N[1], N[2], nchannels])

    def __call__(self, lig_pdb, quantity='all', combine='add'):

        slig = SmallMol(AllChem.MolFromPDBBlock(lig_pdb, sanitize=True, removeHs=False))
        lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, buffer=0, voxelsize=self.voxelsize,
                                                          boxsize=self.boxsize,
                                                          center=self.centers, validitychecks=self.validate,
                                                          method='C')
        if quantity == 'all' and combine == 'add':
            x =self.reshape(lig_vox, lig_N) + self.prot_vox_t
        elif quantity == 'ligand':
            x = self.reshape(lig_vox, lig_N)
        elif quantity == 'protein':
            x = self.prot_vox_t
        else:
            raise ValueError("quantity not good")

        return np.concatenate([x], axis=1)
