from abc import ABC, abstractmethod

from simtk.openmm import app

from rlmm.utils.config import Config


class AbstractSystemLoader(ABC):

    def __init__(self, config_):
        """

        """
        ABC.__init__(self)

    @abstractmethod
    def get_topology(self):
        """

        """
        pass

    @abstractmethod
    def get_positions(self):
        """

        """
        pass

    @abstractmethod
    def get_system(self, **params):
        """

        """
        pass


class PDBSystemLoader(AbstractSystemLoader):
    class Config(Config):
        __slots__ = ['pdb_file_name']

        def __init__(self, config_dict):
            self.pdb_file_name = config_dict['pdb_file_name']

        def get_obj(self):
            return PDBSystemLoader(self)

    def __init__(self, config_: Config):
        super().__init__(config_)
        self.config = config_
        self.pdb = app.PDBFile(self.config.pdb_file_name)
        self.forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    def get_system(self, params):
        """

        :param params:
        :return:
        """
        return self.forcefield.createSystem(topology=self.pdb.topology, **params)

    def get_topology(self):
        """

        :return:
        """
        return self.pdb.topology

    def get_positions(self):
        """

        :return:
        """
        return self.pdb.positions


class AmberSystemLoader(AbstractSystemLoader):
    class Config(Config):
        def __init__(self, resource_root='rlmm/resources/test_adrp_system/'):
            super().__init__()
            self.resource_root = resource_root

    def __init__(self, config: Config):
        """

        :param resource_root:
        """
        super().__init__()
        self.systemloader_config = config
        self.prmtop = app.AmberPrmtopFile(self.systemloader_config.resource_root + 'com.prmtop')
        self.inpcrd = app.AmberInpcrdFile(self.systemloader_config.resource_root + 'com.inpcrd')

    def get_topology(self):
        """

        :return:
        """
        return self.prmtop.topology

    def get_positions(self):
        """

        :return:
        """
        return self.inpcrd.positions

    def get_system(self, params):
        """

        :param params:
        :return:
        """
        return self.prmtop.createSystem(**params)
