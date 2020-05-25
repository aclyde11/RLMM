import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import mdtraj as md
from openeye import oechem, oequacpac
from openforcefield.topology import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from pymol import cmd, stored
from simtk import unit
from simtk.openmm import app

from rlmm.utils.config import Config
from rlmm.utils.loggers import make_message_writer


@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


class PDBLigandSystemBuilder:
    class Config(Config):
        __slots__ = ['pdb_file_name', 'ligand_file_name']

        def update(self, k, v):
            self.__dict__[k] = v

        def __init__(self, config_dict):
            self.tempdir = None
            self.method = config_dict['method']
            self.pdb_file_name = config_dict['pdb_file_name']
            self.ligand_file_name = config_dict['ligand_file_name']
            self.explicit = config_dict['explicit']
            self.config_dict = config_dict

        def get_obj(self):
            return PDBLigandSystemBuilder(self)

    def __init__(self, config_: Config):
        self.config = config_
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.boxvec = None
            self.explicit = self.config.explicit
            self.system = None
            ofs = oechem.oemolistream(self.config.ligand_file_name)
            oemol = oechem.OEMol()
            oechem.OEReadMolecule(ofs, oemol)
            ofs.close()
            self.inital_ligand_smiles = oechem.OEMolToSmiles(oemol)
            self.params_written = 0
            self.mol = Molecule.from_openeye(oemol, allow_undefined_stereo=True)
            fixer = PDBFixer(self.config.pdb_file_name)
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)

            self.config.pdb_file_name = f"{self.config.tempdir(main_context=True)}/inital_fixed.pdb"
            with open(self.config.pdb_file_name, 'w') as f:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
            cmd.reinitialize()
            cmd.load(self.config.pdb_file_name)
            cmd.load(self.config.ligand_file_name, "UNL")
            cmd.alter("UNL", "resn='UNL'")
            cmd.save("{}".format(self.config.pdb_file_name))

    def get_mobile(self):
        return len(self.pdb.positions)

    def __setup_system_ex_warmup_amber(self):
        curr_path = os.getcwd()
        with self.logger("__setup_system_ex_warmup_amber") as logger:
            try:
                with tempfile.TemporaryDirectory() as dirpath:
                    shutil.copy(f'{self.config.tempdir()}apo.pdb', f"{dirpath}/apo.pdb")

                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/apo.pdb')
                    cmd.remove("polymer")
                    cmd.remove("resn HOH or resn Cl or resn Na")
                    cmd.save(f'{dirpath}/lig.pdb')
                    cmd.save(f'{dirpath}/lig.mol2')
                    ifs = oechem.oemolistream(f'{dirpath}/lig.pdb')
                    oemol = oechem.OEMol()
                    oechem.OEReadMolecule(ifs, oemol)
                    ifs.close()
                    ofs = oechem.oemolostream()
                    oemol.SetTitle("UNL")
                    oechem.OEAddExplicitHydrogens(oemol)
                    oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCCharges())
                    if ofs.open(f'{dirpath}/charged.mol2'):
                        oechem.OEWriteMolecule(ofs, oemol)
                    ofs.close()

                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/apo.pdb')
                    cmd.remove("resn UNL or resn UNK")
                    cmd.remove("not polymer")
                    cmd.remove("resn HOH or resn Cl or resn Na")
                    cmd.remove("hydrogens")
                    cmd.save(f'{dirpath}/apo.pdb')

                    with working_directory(dirpath):
                        subprocess.run(
                            f'antechamber -i lig.pdb -fi pdb -o lig.mol2 -fo mol2 -pf y -an y -a charged.mol2 -fa mol2 -ao crg',
                            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        subprocess.run(f'parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod', shell=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        try:
                            subprocess.run(f'pdb4amber -i apo.pdb -o apo_new.pdb --reduce --dry', shell=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except subprocess.CalledProcessError as e:
                            logger.error("Known bug, pdb4amber returns error when there was no error", e.output)
                            pass

                        # Wrap tleap
                        with open('leap.in', 'w+') as leap:
                            leap.write("source leaprc.protein.ff14SB\n")
                            leap.write("source leaprc.water.tip3p \n")
                            leap.write("source leaprc.gaff\n")
                            leap.write("set default PBRadii mbondi3\n")
                            leap.write("rec = loadPDB apo_new.pdb # May need full filepath?\n")
                            leap.write("saveAmberParm rec apo.prmtop apo.inpcrd\n")
                            leap.write("lig = loadmol2 lig.mol2\n")
                            leap.write("loadAmberParams lig.frcmod\n")
                            leap.write("saveAmberParm lig lig.prmtop lig.inpcrd\n")
                            leap.write("com = combine {rec lig}\n")
                            leap.write("saveAmberParm com us_com.prmtop us_com.inpcrd\n")
                            leap.write("solvateBox com TIP3PBOX 10\n")
                            leap.write("saveAmberParm com com.prmtop com.inpcrd\n")
                            leap.write("quit\n")
                        try:
                            subprocess.check_output(f'tleap -f leap.in', shell=True)
                        except subprocess.CalledProcessError as e:
                            logger.error("tleap error", e.output.decode("UTF-8"))
                            exit()

                        prmtop = app.AmberPrmtopFile(f'com.prmtop')
                        inpcrd = app.AmberInpcrdFile(f'com.inpcrd')
                    for comp in ['us_com', 'com', 'apo', 'lig']:
                        for ext in ['prmtop', 'inpcrd']:
                            shutil.copy(f'{dirpath}/{comp}.{ext}',
                                        f"{self.config.tempdir()}com_{self.params_written}.{ext}")
                    system = prmtop.createSystem(**self.params)
                    topology, positions = prmtop.topology, inpcrd.positions

                    _topology = md.Topology.from_openmm(topology)
                    cs = 0
                    for i, atom in enumerate(_topology.atoms):
                        if atom.residue.name.lower() in ['hoh', 'cl', 'na']:
                            continue  # Skip these atoms
                        cs += 1
                        system.setParticleMass(i, 0 * unit.dalton)
                    self.params_written += 1
                    return system, topology, positions
            except Exception as e:
                logger.error("EXCEPTION CAUGHT BAD SPOT", e.output.decode("UTF-8"))

    def __setup_system_ex_warmup_mm(self):
        with self.logger("__setup_system_ex_warmup_mm") as logger:
            amber_forcefields = ['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml']
            small_molecule_forcefield = 'openff-1.1.0'
            # small_molecule_forcefield = 'gaff-2.11'

            openmm_system_generator = SystemGenerator(forcefields=amber_forcefields,
                                                      forcefield_kwargs=self.warmupparams,
                                                      molecules=[self.mol],
                                                      small_molecule_forcefield=small_molecule_forcefield,
                                                      )

            boxvec = self.boxvec
            system = openmm_system_generator.create_system(self.topology)
            system.setDefaultPeriodicBoxVectors(*boxvec)

            topology = md.Topology.from_openmm(self.topology)
            cs = 0
            for i, atom in enumerate(topology.atoms):
                if atom.residue.name.lower() in ['hoh', 'cl', 'na']:
                    continue  # Skip these atoms
                cs += 1
                system.setParticleMass(i, 0 * unit.dalton)

        return system, self.topology, self.positions

    def __setup_system_ex_mm(self):
        with self.logger("__setup_system_ex_mm") as logger:
            if "openmm_system_generator" not in self.__dict__:
                amber_forcefields = ['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml']
                small_molecule_forcefield = 'openff-1.1.0'
                # small_molecule_forcefield = 'gaff-2.11'
                self.openmm_system_generator = SystemGenerator(forcefields=amber_forcefields,
                                                               forcefield_kwargs=self.params,
                                                               molecules=[self.mol],
                                                               small_molecule_forcefield=small_molecule_forcefield,
                                                               )

            else:
                self.openmm_system_generator.add_molecules([self.mol])

            self.modeller = app.Modeller(self.topology, self.positions)
            self.modeller.addSolvent(self.openmm_system_generator.forcefield, model='tip3p',
                                     ionicStrength=100 * unit.millimolar, padding=1.0 * unit.nanometers)
            self.boxvec = self.modeller.getTopology().getPeriodicBoxVectors()
            self.topology, self.positions = self.modeller.getTopology(), self.modeller.getPositions()
            self.system = self.openmm_system_generator.create_system(self.topology)
            self.system.setDefaultPeriodicBoxVectors(*self.modeller.getTopology().getPeriodicBoxVectors())

            with open("{}".format(self.config.pdb_file_name), 'w') as f:
                app.PDBFile.writeFile(self.topology, self.positions, file=f, keepIds=True)
                logger.log("wrote ", "{}".format(self.config.pdb_file_name))
            with open("{}".format(self.config.pdb_file_name), 'r') as f:
                self.pdb = app.PDBFile(f)
        return self.system, self.topology, self.positions

    def __setup_system_ex_amber(self, pdbfile: str = None):
        with self.logger("__setup_system_ex_amber") as logger:
            try:
                with tempfile.TemporaryDirectory() as dirpath:
                    dirpath = self.config.tempdir()

                    # Move inital file over to new system.
                    shutil.copy(pdbfile, f"{dirpath}/init.pdb")

                    # Assign charges and extract new ligand
                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/init.pdb')
                    cmd.remove("polymer")
                    cmd.remove("resn HOH or resn Cl or resn Na")
                    cmd.save(f'{dirpath}/lig.pdb')
                    cmd.save(f'{dirpath}/lig.mol2')
                    ifs = oechem.oemolistream(f'{dirpath}/lig.pdb')
                    oemol = oechem.OEMol()
                    oechem.OEReadMolecule(ifs, oemol)
                    ifs.close()
                    ofs = oechem.oemolostream()
                    oemol.SetTitle("UNL")
                    oechem.OEAddExplicitHydrogens(oemol)
                    oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCCharges())
                    if ofs.open(f'{dirpath}/charged.mol2'):
                        oechem.OEWriteMolecule(ofs, oemol)
                    ofs.close()

                    # remove hydrogens and ligand from PDB
                    cmd.reinitialize()
                    cmd.load(f'{dirpath}/init.pdb')
                    cmd.remove("not polymer")
                    cmd.remove("hydrogens")
                    cmd.save(f'{dirpath}/apo.pdb')

                    with working_directory(dirpath):
                        subprocess.run(
                            f'antechamber -i lig.pdb -fi pdb -o lig.mol2 -fo mol2 -pf y -an y -a charged.mol2 -fa mol2 -ao crg'.split(
                                " "), check=True, capture_output=True)
                        subprocess.run(f'parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod'.split(" "), check=True,
                                       capture_output=True)
                        try:
                            subprocess.run('pdb4amber -i apo.pdb -o apo_new.pdb --reduce --dry'.split(" "), check=True,
                                           capture_output=True)
                        except subprocess.CalledProcessError as e:
                            logger.error("Known bug, pdb4amber returns error when there was no error", e.stdout, e.stderr)
                            pass

                        # Wrap tleap
                        with open('leap.in', 'w+') as leap:
                            leap.write("source leaprc.protein.ff14SB\n")
                            leap.write("source leaprc.water.tip3p \n")
                            leap.write("source leaprc.gaff\n")
                            leap.write("set default PBRadii mbondi3\n")
                            leap.write("rec = loadPDB apo_new.pdb # May need full filepath?\n")
                            leap.write("saveAmberParm rec apo.prmtop apo.inpcrd\n")
                            leap.write("lig = loadmol2 lig.mol2\n")
                            leap.write("loadAmberParams lig.frcmod\n")
                            leap.write("saveAmberParm lig lig.prmtop lig.inpcrd\n")
                            leap.write("com = combine {rec lig}\n")
                            leap.write("saveAmberParm com us_com.prmtop us_com.inpcrd\n")
                            leap.write("solvateBox com TIP3PBOX 10\n")
                            leap.write("saveAmberParm com com.prmtop com.inpcrd\n")
                            leap.write("quit\n")
                        try:
                            subprocess.run('tleap -f leap.in'.split(" "), check=True, capture_output=True)
                        except subprocess.CalledProcessError as e:
                            logger.error("tleap error", e.output.decode("UTF-8"))
                            exit()

                        prmtop = app.AmberPrmtopFile(f'com.prmtop')
                        inpcrd = app.AmberInpcrdFile(f'com.inpcrd')

                    for comp in ['us_com', 'com', 'apo', 'lig']:
                        for ext in ['prmtop', 'inpcrd']:
                            shutil.copy(f'{dirpath}/{comp}.{ext}',
                                        f"{self.config.tempdir()}/{comp}_{self.params_written}.{ext}")

                    self.system = prmtop.createSystem(**self.params)
                    self.boxvec = self.system.getDefaultPeriodicBoxVectors()
                    self.topology, self.positions = prmtop.topology, inpcrd.positions
                    with open("{}".format(self.config.pdb_file_name), 'w') as f:
                        app.PDBFile.writeFile(self.topology, self.positions, file=f, keepIds=True)
                        logger.log("wrote ", "{}".format(self.config.pdb_file_name))
                    with open("{}".format(self.config.pdb_file_name), 'r') as f:
                        self.pdb = app.PDBFile(f)
                    self.params_written += 1

                    return self.system, self.topology, self.positions
            except Exception as e:
                logger.error("EXCEPTION CAUGHT BAD SPOT", e)

    def __setup_system_im(self, pdbin=None):
        with self.logger("__setup_system_im") as logger:
            try:
                with tempfile.TemporaryDirectory() as dirpath:
                    if pdbin is None:
                        pdbin = f'{self.config.tempdir}apo.pdb'
                    shutil.copy(pdbin, f"{dirpath}/apo.pdb")

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
                    oemol.SetTitle("UNL")
                    oechem.OEAddExplicitHydrogens(oemol)
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
                        subprocess.run(f'parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod', shell=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        try:
                            subprocess.run(f'pdb4amber -i apo.pdb -o apo_new.pdb --reduce --dry', shell=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except subprocess.CalledProcessError as e:
                            logger.error("Known bug, pdb4amber returns error when there was no error", e.output)
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
                            logger.error("tleap error", e.output.decode("UTF-8"))
                            exit()

                        prmtop = app.AmberPrmtopFile(f'com.prmtop')
                        inpcrd = app.AmberInpcrdFile(f'com.inpcrd')

                    for comp in ['com', 'apo', 'lig']:
                        for ext in ['prmtop', 'inpcrd']:
                            shutil.copy(f'{dirpath}/{comp}.{ext}',
                                        f"{self.config.tempdir}{comp}_{self.params_written}.{ext}")
                    self.system = prmtop.createSystem(**self.params)
                    self.topology, self.positions = prmtop.topology, inpcrd.positions
                    return self.system, self.topology, self.positions
            except Exception as e:
                logger.error("EXCEPTION CAUGHT BAD SPOT", e.output.decode("UTF-8"))

    def get_warmup_system(self, params):
        """

        :param params:
        :return:
        """
        with self.logger("get_system") as logger:
            self.warmupparams = params

            logger.log("Loading inital system", self.config.pdb_file_name)
            if self.config.explicit and self.config.method == 'amber':
                return self.__setup_system_ex_warmup_amber()
            elif self.config.explicit:
                return self.__setup_system_ex_warmup_mm()
            else:
                assert (False)

    def get_system(self, params):
        """

        :param params:
        :return:
        """
        with self.logger("get_system") as logger:
            self.params = params

            logger.log("Loading inital system", self.config.pdb_file_name)
            self.pdb = app.PDBFile(self.config.pdb_file_name)
            self.topology, self.positions = self.pdb.topology, self.pdb.positions

            if self.config.explicit and self.config.method == 'amber':
                self.system, self.topology, self.positions = self.__setup_system_ex_amber(
                    pdbfile=self.config.pdb_file_name)
            elif self.config.explicit:
                self.system, self.topology, self.positions = self.__setup_system_ex_mm()
            else:
                self.system, self.topology, self.positions = self.__setup_system_im()

        return self.system

    def reload_system(self, ln: str, smis: oechem.OEMol, old_pdb: str, is_oe_already: bool = False):
        with self.logger("reload_system") as logger:
            logger.log("Loading {} with new smiles {}".format(old_pdb, ln))
            with tempfile.TemporaryDirectory() as dirpath:
                # self.mol = Molecule.from_smiles(ln, hydrogens_are_explicit=True, allow_undefined_stereo=True)

                ofs = oechem.oemolostream("{}/newlig.mol2".format(dirpath))
                oechem.OEWriteMolecule(ofs, smis)
                ofs.close()
                cmd.reinitialize()
                cmd.load(old_pdb)
                cmd.remove("not polymer")
                cmd.load("{}/newlig.mol2".format(dirpath), "UNL")
                cmd.alter("UNL", "resn='UNL'")
                cmd.alter("UNL", "chain='A'")
                self.config.pdb_file_name = self.config.tempdir() + "reloaded.pdb"
                cmd.save(self.config.pdb_file_name)
                cmd.save(self.config.tempdir() + "apo.pdb")

                with open(self.config.pdb_file_name, 'r') as f:
                    self.pdb = app.PDBFile(f)
                self.positions, self.topology = self.pdb.getPositions(), self.pdb.getTopology()

                if self.config.explicit and self.config.method == 'amber':
                    self.system, self.topology, self.positions = self.__setup_system_ex_amber(
                        pdbfile=self.config.pdb_file_name)
                elif self.config.explicit:
                    self.system, self.topology, self.positions = self.__setup_system_ex_mm()
                else:
                    self.system, self.topology, self.positions = self.__setup_system_im()

        return self.system

    def get_selection_ids(self, select_cmd):
        with tempfile.TemporaryDirectory() as dirname:
            with open(f'{dirname}/get_selection_ids.pdb', 'w') as f:
                app.PDBFile.writeFile(self.get_topology(),
                                      self.get_positions(),
                                      file=f)
            cmd.reinitialize()
            cmd.load(f'{dirname}/get_selection_ids.pdb', format='pdb')
            cmd.select("sele", select_cmd)
            stored.ids = list()
            cmd.iterate("sele", expression="stored.ids.append(ID)")
            ids = [int(i - 1) for i in list(stored.ids)]
        return ids

    def get_selection_solvent(self):
        ids = [i - 2 for i in self.get_selection_ids("not polymer and not (resn UNL)")]
        if len(ids) == 0:
            return []
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_solvent", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_selection_ligand(self):
        ids = [i for i in self.get_selection_ids("resn UNL")]
        if len(ids) == 0:
            return []
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_ligand", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_selection_protein(self):
        ids = self.get_selection_ids("polymer")
        if len(ids) == 0:
            return []
        if not ((min(ids) >= 0) and (max(ids) < len(self.positions))):
            self.logger.static_failure("get_selection_protein", min(ids), max(ids), len(self.positions), exit_all=True)
        return ids

    def get_topology(self):
        return self.topology

    def get_positions(self):
        return self.positions
