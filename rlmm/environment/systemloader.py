import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from simtk import unit
from openeye import oechem, oequacpac
from openforcefield.topology import Molecule
from openmmforcefields.generators import SystemGenerator
import subprocess
import os, contextlib, sys
from pdbfixer import PDBFixer
from pymol import cmd, stored
from simtk.openmm import app
from rlmm.utils.loggers import make_message_writer

from rlmm.utils.config import Config


@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


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

        def update(self, k, v):
            self.__dict__[k] = v

        def __init__(self, config_dict):
            self.pdb_file_name = config_dict['pdb_file_name']
            self.ligand_file_name = config_dict['ligand_file_name']
            self.config_dict = config_dict

        def get_obj(self):
            return PDBLigandSystemBuilder(self)

    def __init__(self, config_: Config):
        self.config = config_
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            super().__init__(config_)

            ofs = oechem.oemolistream(self.config.ligand_file_name)
            oemol = oechem.OEMol()
            oechem.OEReadMolecule(ofs, oemol)
            ofs.close()
            self.inital_ligand_smiles = oechem.OEMolToSmiles(oemol)

            fixer = PDBFixer(self.config.pdb_file_name)
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)

            self.config.pdb_file_name = self.config.tempdir + "inital_fixed.pdb"
            with open(self.config.pdb_file_name, 'w') as f:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
            cmd.reinitialize()
            cmd.load(self.config.pdb_file_name)
            cmd.load(self.config.ligand_file_name, "UNL")
            cmd.alter("UNL", "resn='UNL'")
            cmd.save("{}".format(self.config.pdb_file_name))

    def get_mobile(self):
        return len(self.pdb.positions)

    # def __setup_system_ex(self, oemol : oechem.OEMolBase = None, lig_mol : =None):
    #     amber_forcefields = ['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml',
    #                               'amber/tip3p_HFE_multivalent.xml']
    #     small_molecule_forcefield = 'gaff-2.11'
    #     openmm_system_generator = SystemGenerator(forcefields=amber_forcefields,
    #                                               forcefield_kwargs=self.params,
    #                                               nonperiodic_forcefield_kwargs=self.params,
    #                                               molecules=[self.mol],
    #                                               small_molecule_forcefield=small_molecule_forcefield,
    #                                               )
    #
    #     self.topology, self.positions = self.pdb.topology, self.pdb.positions
    #     modeller = app.Modeller(self.topology, self.positions)
    #     modeller.addSolvent(openmm_system_generator.forcefield, padding=1.0 * unit.nanometers)
    #     self.system = openmm_system_generator.create_system(modeller.topology)
    #     self.topology, self.positions = modeller.topology, modeller.positions

    def __setup_system_im(self, oemol : oechem.OEMolBase =None, lig_mol=None):
        #TODO Austin is this
        with self.logger("__setup_system_im") as logger:
            with tempfile.TemporaryDirectory() as dirpath:
                app.Modeller(self.topology, self.positions)
                with open(f'{dirpath}/apo.pdb', 'w') as f:
                    app.PDBFile.writeFile(self.get_topology(),
                                          self.get_positions(),
                                          file=f, keepIds=True)

                if lig_mol is not None and oemol is None:
                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/apo.pdb')
                    cmd.remove("polymer")
                    cmd.save(f'{dirpath}/lig.pdb')
                    cmd.save(f'{dirpath}/lig.mol2')
                    ifs = oechem.oemolistream(f'{dirpath}/lig.pdb')
                    oemol = oechem.OEMol()
                    oechem.OEReadMolecule(ifs, oemol)
                    ifs.close()
                    ofs = oechem.oemolostream()
                    oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCCharges())
                    oechem.OEAddExplicitHydrogens(oemol)
                    if ofs.open(f'{dirpath}/charged.mol2'):
                        oechem.OEWriteMolecule(ofs, oemol)
                    ofs.close()
                else:
                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/apo.pdb')
                    cmd.remove("polymer")
                    cmd.save(f'{dirpath}/lig.pdb')
                    cmd.save(f'{dirpath}/lig.mol2')
                    ofs = oechem.oemolostream()
                    oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCCharges())
                    if ofs.open(f'{dirpath}/charged.mol2'):
                        oechem.OEWriteMolecule(ofs, oemol)
                    ofs.close()
                cmd.reinitialize()
                cmd.load(f'{dirpath}/apo.pdb')
                cmd.remove("resn UNL or resn UNK")
                cmd.remove("not polymer")
                cmd.remove("hydrogens")
                cmd.save(f'{dirpath}/apo.pdb')

                with working_directory(dirpath):
                    subprocess.run(
                        f'antechamber -i lig.pdb -fi pdb -o lig.mol2 -fo mol2 -pf y -an y -a charged.mol2 -fa mol2 -ao crg',
                        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(f'parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        subprocess.run(f'pdb4amber -i apo.pdb -o apo_new.pdb --reduce --dry', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except subprocess.CalledProcessError as e:
                        logger.log("Known bug, pdb4amber returns error when there was no error", e.output)
                        pass

                    # Wrap tleap
                    with open('leap.in', 'w+') as leap:
                        leap.write("source leaprc.protein.ff14SBonlysc\n")
                        leap.write("source leaprc.gaff\n")
                        leap.write("set default PBRadii mbondi3\n")
                        leap.write("rec = loadPDB apo_new.pdb # May need full filepath?\n")
                        leap.write("saveAmberParm rec apo.prmtop apo.inpcrd\n")
                        leap.write("lig = loadmol2 lig.mol2\n")
                        leap.write("loadAmberParams lig.frcmod\n")
                        leap.write("com = combine {rec lig}\n")
                        leap.write("saveAmberParm lig lig.prmtop lig.inpcrd\n")
                        leap.write("saveAmberParm com com.prmtop com.inpcrd\n")
                        leap.write("quit\n")
                    try:
                        subprocess.check_output(f'tleap -f leap.in', shell=True)
                    except subprocess.CalledProcessError as e:
                        logger.log("tleap error", e.output)
                        pass

                    prmtop = app.AmberPrmtopFile(f'com.prmtop')
                    inpcrd = app.AmberInpcrdFile(f'com.inpcrd')
                    self.system = prmtop.createSystem(**self.params)
                    self.topology, self.positions = prmtop.topology, inpcrd.positions

    def get_system(self, params, explict=False):
        """

        :param params:
        :return:
        """
        with self.logger("get_system") as logger:
            self.params = params

            logger.log("Loading inital system", self.config.pdb_file_name)
            self.pdb = app.PDBFile(self.config.pdb_file_name)
            self.topology, self.positions = self.pdb.topology, self.pdb.positions
            if not explict:
                self.__setup_system_im(lig_mol=self.config.ligand_file_name)
            else:
                self.__setup_system_ex(lig_mol=self.config.ligand_file_name)

        return self.system

    def reload_system(self, ln : str , smis : oechem.OEMolBase, old_pdb : str, is_oe_already : bool = False, explict : bool =False):
        with self.logger("reload_system") as logger:
            logger.log("Loading {} with new smiles {}".format(old_pdb, ln))
            with tempfile.TemporaryDirectory() as dirpath:
                ofs = oechem.oemolostream("{}/newlig.sdf".format(dirpath))
                oechem.OEWriteMolecule(ofs, smis)
                ofs.close()
                cmd.reinitialize()
                cmd.load(old_pdb)
                cmd.remove("not polymer")
                cmd.load("{}/newlig.sdf".format(dirpath), 'UNL')
                cmd.alter("UNL", "resn='UNL'")
                self.config.pdb_file_name = self.config.tempdir + "reloaded.pdb"
                cmd.save("{}".format(self.config.pdb_file_name))
            self.pdb = app.PDBFile(self.config.pdb_file_name)
            self.topology, self.positions = self.pdb.topology, self.pdb.positions
            self.mol = Molecule.from_openeye(smis, allow_undefined_stereo=True)
            self.__setup_system_im(oemol=smis)

        return self.system

    def get_selection_ids(self, select_cmd):
        with tempfile.NamedTemporaryFile('w') as f:
            app.PDBFile.writeFile(self.get_topology(),
                                  self.get_positions(),
                                  file=f)
            cmd.reinitialize()
            cmd.load(f.name, object="pdb", format='pdb')
            cmd.select("sele", select_cmd)
            stored.ids = list()
            cmd.iterate("sele", expression="stored.ids.append(ID)")
            ids = [int(i - 1) for i in list(stored.ids)]
        return ids

    def get_selection_solvent(self):
        ids = [i - 2 for i in self.get_selection_ids("not polymer and not (resn UNK or resn UNL)")]
        if len(ids) == 0:
            return []
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_solvent", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_selection_ligand(self):
        ids = [i for i in self.get_selection_ids("resn UNK or resn UNL")]
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_ligand", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_selection_protein(self):
        ids = self.get_selection_ids("polymer")
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_protein", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_topology(self):
        return self.topology

    def get_positions(self):
        return self.positions


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

    # TODO: default just move everything around, but this needs to param on ligand.
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
