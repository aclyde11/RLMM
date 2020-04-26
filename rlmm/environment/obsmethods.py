from abc import ABC, abstractmethod

import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from rdkit.Chem import AllChem

from rlmm.utils.config import Config
import cv2


class AbstractObsMethod(ABC):

    def __init__(self, obs_config: Config):
        """

        """
        super().__init__()

    @abstractmethod
    def from_simulation(self, simulation):
        """

        :param simulation:
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """

        """
        pass


class CoordinatePCA(AbstractObsMethod):
    class Config(Config):
        def __init__(self, config_dict):
            pass

        def get_obj(self):
            return CoordinatePCA(self)

    def __init__(self, obs_config: Config):
        """
            Generates PCA image of coordinates. This is just a dummy system for testing, not recommended for use.
        """
        super().__init__(obs_config)

    def from_simulation(self, simulation):
        """

        :param simulation:
        :return:
        """
        return self(simulation.get_coordinates())

    def __call__(self, coordinates):
        """

        :param coordinates: image of PCA plot as a numpy array in np.float32
        :return:
        """
        from sklearn.decomposition import PCA
        from matplotlib import pyplot as plt
        import io
        from PIL import Image

        pca = PCA(2)
        fit = pca.fit_transform(coordinates)
        plt.scatter(fit[:, 0], fit[:, 1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = np.asarray(Image.open(buf),dtype=np.float32)
        im = cv2.resize(im, (24,32))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        buf.close()
        return im


class Voxelizer(AbstractObsMethod):
    class Config(Config):
        def __init__(self):
            self.boxsize = None
            self.centers = None
            self.voxelsize = None
            self.validity_check = False
            self.method = 'C'
            self.channel_first = False
            self.pdb = None
            self.quantity = 'all'
            self.combine = 'add'
            super().__init__()

    def __init__(self, obs_config: Config):
        """

        :param obs_config
        """
        super().__init__(obs_config)
        self.obs_config = obs_config

        prot = Molecule(self.obs_config.pdb)

        prot = prepareProteinForAtomtyping(prot, verbose=False)
        prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, buffer=0, voxelsize=self.obs_config.voxelsize,
                                                             boxsize=self.obs_config.boxsize,
                                                             center=self.obs_config.centers,
                                                             method=self.obs_config.method,
                                                             validitychecks=self.obs_config.validity_check)

        self.prot_vox_t = self.reshape(prot_vox, prot_N)

    def reshape(self, x, N):
        """

        :param x:
        :param N:
        :return:
        """
        nchannels = x.shape[1]

        x = x.tranpose()
        if self.obs_config.channel_first:
            return x.transpose().reshape([1, nchannels, N[0], N[1], N[2]])
        else:
            return x.transpose().reshape([1, N[0], N[1], N[2], nchannels])

    def from_simulation(self, simulation):
        """

        :param simulation:
        :return:
        """
        return self(simulation.get_pdb())

    def __call__(self, lig_pdb):
        """

        :param lig_pdb:
        :return:
        """
        slig = SmallMol(AllChem.MolFromPDBBlock(lig_pdb, sanitize=True, removeHs=False))
        lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, buffer=0, voxelsize=self.obs_config.voxelsize,
                                                          boxsize=self.obs_config.boxsize,
                                                          center=self.obs_config.centers, method=self.obs_config.method,
                                                          validitychecks=self.obs_config.validity_check)
        if self.obs_config.quantity == 'all' and self.obs_config.combine == 'add':
            x = self.reshape(lig_vox, lig_N) + self.prot_vox_t
        elif self.obs_config.quantity == 'ligand':
            x = self.reshape(lig_vox, lig_N)
        elif self.obs_config.quantity == 'protein':
            x = self.prot_vox_t
        else:
            raise ValueError("quantity not good")

        return np.concatenate([x], axis=1)
