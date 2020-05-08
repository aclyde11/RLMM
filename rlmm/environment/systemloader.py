from abc import ABC, abstractmethod
from simtk import openmm, unit
from simtk.openmm import app

from rlmm.utils.config import Config
from openforcefield.topology import Molecule
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import rdmolops
from openmmforcefields.generators import SystemGenerator

from pymol import cmd

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

class PDBLigandSystemBuilder(AbstractSystemLoader):
    class Config(Config):
        __slots__ = ['pdb_file_name', 'ligand_file_name']

        def __init__(self, config_dict):
            self.pdb_file_name = config_dict['pdb_file_name']
            self.ligand_file_name = config_dict['ligand_file_name']
            self.ligand_smiles = config_dict['ligand_smiles']

        def get_obj(self):
            return PDBLigandSystemBuilder(self)

    def __init__(self, config_: Config):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        super().__init__(config_)
        self.config = config_

        fixer = PDBFixer(self.config.pdb_file_name)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        self.config.pdb_file_name = self.config.pdb_file_name.split(".")[0] + "_fixed.pdb"
        with open(self.config.pdb_file_name, 'w') as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions,f)

        cmd.reinitialize()
        cmd.load(self.config.pdb_file_name)
        cmd.load(self.config.ligand_file_name, "UNL")
        cmd.do("alter UNL, resn='UNL'")
        self.config.pdb_file_name = self.config.pdb_file_name.split(".")[0] + "_com.pdb"
        cmd.do("save {}".format(self.config.pdb_file_name))

    def get_mobile(self):
        return len(self.pdb.positions)

    def get_system(self, params):
        """

        :param params:
        :return:
        """
        self.pdb = app.PDBFile(self.config.pdb_file_name)
        self.mol = Molecule.from_smiles(self.config.ligand_smiles)

        protein_forcefield = 'amber14/protein.ff14SB.xml'
        small_molecule_forcefield = 'openff-1.1.0'
        solvation_forcefield = 'amber14/tip3p.xml'

        forcefields = [protein_forcefield, solvation_forcefield]
        openmm_system_generator = SystemGenerator(forcefields=forcefields,
                                                  molecules=[self.mol],
                                                  small_molecule_forcefield=small_molecule_forcefield)

        topology, positions = self.pdb.topology, self.pdb.positions
        modeller = app.Modeller(topology, positions)
        self.system = openmm_system_generator.create_system(modeller.topology)
        return self.system

    def reload_system(self, ln,smis, old_pdb):
        import openforcefield.utils
        from openeye import oechem
        ofs = oechem.oemolostream("test2.sdf")
        oechem.OEWriteMolecule(ofs, smis)
        ofs.close()

        cmd.reinitialize()
        cmd.load(old_pdb)
        cmd.remove("resn UNL")
        cmd.load('test2.sdf', 'UNL')
        cmd.do("alter UNL, resn='UNL'")
        cmd.do("alter UNL, resid='1'")
        self.config.pdb_file_name = self.config.pdb_file_name.split(".")[0] + "_com.pdb"
        cmd.do("save {}".format(self.config.pdb_file_name))
        self.pdb = app.PDBFile(self.config.pdb_file_name)

        self.mol = Molecule.from_openeye(smis, allow_undefined_stereo=True)
        protein_forcefield = 'amber14/protein.ff14SB.xml'
        # small_molecule_forcefield = 'gaff-2.1'
        small_molecule_forcefield = 'openff-1.1.0'
        solvation_forcefield = 'amber14/tip3p.xml'


        forcefields = [protein_forcefield, solvation_forcefield]
        openmm_system_generator = SystemGenerator(forcefields=forcefields,
                                                  molecules=[self.mol],
                                                  small_molecule_forcefield=small_molecule_forcefield,
                                                  forcefield_kwargs={'allow_nonintegral_charges' : True})
        # openmm_system_generator.add_molecules(self.mol)
        # openmm_system_generator.add_molecules([self.mol,self.mol2])
        topology, positions = self.pdb.topology, self.pdb.positions
        modeller = app.Modeller(topology, positions)
        print(openmm_system_generator.forcefield)
        print(self.config.pdb_file_name)
        self.system = openmm_system_generator.create_system(modeller.topology)
        # except openforcefield.utils.toolkits.UndefinedStereochemistryError:
        #     self.mol = Molecule.from_openeye(smis, allow_undefined_stereo=True)
        #
        #     protein_forcefield = 'amber14/protein.ff14SB.xml'
        #     # small_molecule_forcefield = 'gaff-2.1'
        #     small_molecule_forcefield = 'openff-1.1.0'
        #     solvation_forcefield = 'amber14/tip3p.xml'
        #
        #     forcefields = [protein_forcefield, solvation_forcefield]
        #     openmm_system_generator = SystemGenerator(forcefields=forcefields,
        #                                               molecules=[self.mol],
        #                                               small_molecule_forcefield=small_molecule_forcefield)
        #     # openmm_system_generator.add_molecules(self.mol)
        #     # openmm_system_generator.add_molecules([self.mol,self.mol2])
        #     topology, positions = self.pdb.topology, self.pdb.positions
        #     modeller = app.Modeller(topology, positions)
        #     print(openmm_system_generator.forcefield)
        #     print(self.config.pdb_file_name)
        #     self.system = openmm_system_generator.create_system(modeller.topology)
        return self.system

    def get_selection_pos(self, selection):
        from io import StringIO
        output = StringIO()
        app.PDBFile.writeFile(self.get_topology(),
                              self.get_positions(),
                              file=output)
        pdb = output.getvalue()

        poss = []
        for line in pdb.split("\n"):
            if "UNL" in line and (not 'TER' in line):
                posn = int(line.split(" ")[1])
                print(posn)
                poss.append(posn - 1)

        return poss


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

    #TODO: default just move everything around, but this needs to param on ligand.
    def get_mobile(self):
        return len(self.pdb.positions)

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
