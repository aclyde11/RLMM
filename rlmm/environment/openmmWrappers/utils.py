import itertools
import subprocess
from io import StringIO
import shutil
import mdtraj as md
from simtk import openmm as mm
import mdtraj.utils as mdtrajutils
import numpy as np
import openmmtools
import pandas as pd
from openmmtools.mcmc import WeightedMove, LangevinSplittingDynamicsMove, SequenceMove, \
    MCDisplacementMove, MCRotationMove, GHMCMove
from simtk import unit
from simtk.openmm import app

from rlmm.environment.systemloader import working_directory
from rlmm.utils.config import Config


def get_mdtraj_box(boxvec=None, a=None, b=None, c=None, iterset=-1):
    if boxvec is not None:
        a, b, c = boxvec
    elif None in [a, b, c]:
        exit()

    a, b, c = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom), c.value_in_unit(unit.angstrom)
    a, b, c = np.array(a), np.array(b), np.array(c)
    a, b, c, alpha, beta, gamma = md.utils.unitcell.box_vectors_to_lengths_and_angles(a, b, c)
    if iterset is -1:
        return a, b, c, alpha, beta, gamma
    else:
        return [[a, b, c]] * iterset, [[alpha, beta, gamma]] * iterset


def get_coordinates_samplers(topology: app.topology.Topology, sampler_state: openmmtools.states.SamplerState, explicit: bool):
    return get_coordinates(topology, sampler_state.positions, sampler_state.box_vectors, sampler_state.n_particles,
                           explicit)


def get_coordinates(topology: app.topology.Topology,
                    coords,
                    boxvecs,
                    n_particles: int,
                    explicit: bool):
    coords = np.array(coords.value_in_unit(unit.angstrom))

    if explicit:
        trajectory_positions = coords.reshape(
            (1, n_particles, 3))
        lengths, angles = get_mdtraj_box(boxvec=boxvecs, iterset=1)

        traj = md.Trajectory(trajectory_positions, md.Topology.from_openmm(topology),
                             unitcell_lengths=lengths, unitcell_angles=angles)

        traj = traj.image_molecules(inplace=False)
        coords = traj.xyz.reshape((traj.n_atoms, 3))

    return coords


def get_pdb(topology, coords, file_name=None):
    if file_name is None:
        output = StringIO()
    else:
        output = open(file_name, 'w')
    app.PDBFile.writeFile(topology,
                          coords,
                          file=output, keepIds=False)
    if file_name is None:
        return output.getvalue()
    else:
        output.close()
        return True

def get_ligand_ids(topology):
    return md.Topology.from_openmm(topology).select("resn UNL")

def get_protein_ids(topology):
    return md.Topology.from_openmm(topology).select("protein")

def get_backbone_ids(topology):
    return md.Topology.from_openmm(topology).select("protein and backbone")

def get_selection_ids(topology, sele):
    return md.Topology.from_openmm(topology).select(sele)

def get_backbone_restraint_force(topology, positions, explicit, K=5.0):
    if explicit:
        energy_expression = '(k_restr/2)*periodicdistance(x, y, z, x0, y0, z0)^2' # periodic distance
    else:
        energy_expression = '(k_restr/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)' # non-periodic distance

    force = mm.CustomExternalForce(energy_expression)
    force.addGlobalParameter("k_restr", K)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    positions_ = positions.value_in_unit(unit.nanometer)
    for i, atom_id in enumerate(get_backbone_ids(topology)):
        pos = positions_[atom_id]
        pops = mm.Vec3(pos[0], pos[1], pos[2])
        _ = force.addParticle(int(atom_id), pops)
    return force

def get_ligand_restraint_force(topology, positions, explicit, K=5.0):
    if explicit:
        energy_expression = '(k_restr/2)*periodicdistance(x, y, z, x0, y0, z0)^2' # periodic distance
    else:
        energy_expression = '(k_restr/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)' # non-periodic distance

    force = mm.CustomExternalForce(energy_expression)
    force.addGlobalParameter("k_restr", K)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    positions_ = positions.value_in_unit(unit.nanometer)
    for i, atom_id in enumerate(get_ligand_ids(topology)):
        pos = positions_[atom_id]
        pops = mm.Vec3(pos[0], pos[1], pos[2])
        _ = force.addParticle(int(atom_id), pops)
    return force

def detect_ligand_flyaway(traj, eps=2.0):
    traj = traj.atom_slice(traj.topology.select("protein or resn UNL"))
    resn = len(list(traj.topology.residues))
    group_1 = list(range(resn))
    group_2 = [resn - 1]
    pairs = list(itertools.product(group_1, group_2))
    res, pairs = md.compute_contacts(traj, pairs)
    pocket_resids = list(np.where(res[0] <= 5)[0] + 1)
    pocket_resids = ["resid {}".format(id) for id in pocket_resids]
    pocket_resids = " or ".join(pocket_resids)

    group_1 = list(traj.topology.select(pocket_resids))
    group_2 = list(traj.topology.select("not protein"))
    pairs = list(itertools.product(group_1, group_2))
    res = md.compute_distances(traj, pairs)
    distances = np.quantile(res, 0.5, axis=1)
    if np.abs(np.mean(distances[:int(distances.shape[0] * 0.1)]) - np.mean(distances[int(distances.shape[0]* 0.9):])) >= eps:
        return True
    else:
        return False

def run_amber_mmgbsa(logger, explicit, tempdir, run_decomp=False):

    with logger('run_amber_mmgbsa') as logger:
        logger.log("PASSING")
        return
        complex_prmtop = f"com.prmtop"
        traj = "traj.dcd"
        with working_directory(tempdir):
            if explicit:
                with open("cpptraj_input.txt", 'w') as f:
                    f.write("strip :WAT parmout stripped.prmtop outprefix traj.dcd nobox\n" +
                            "trajout test2.dcd\n" +
                            "run\n")
                subprocess.run(['cpptraj', '-p', complex_prmtop, '-y', traj, '-i', 'cpptraj_input.txt'],
                                      check=True, capture_output=True)
                complex_prmtop = "stripped.prmtop"
                traj = "test2.dcd"
            else:
                shutil.copyfile(com)

            try:
                subprocess.run(['ante-MMPBSA.py',
                                       '-p', complex_prmtop,
                                       '-l', 'noslig.prmtop',
                                       '-r', 'nosapo.prmtop',
                                       '-n', ':UNL'], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(e, e.stderr.decode("UTF-8"), e.stdout.decode("UTF-8"))

            with open("mmpbsa_input.txt", 'w') as f:
                f.write(
                    '&general\ninterval=5,\nverbose=3, keep_files=0, strip_mask=":WAT:CL:CIO:CS:IB:K:LI:MG:NA:RB:HOH",\n/\n&gb\nigb=5, saltcon=0.1000,\n/\n'
                )
                if run_decomp:
                    f.write('&decomp\nidecomp=1,csv_format=1\n/\n')

            logger.log("Running amber MMPBSA.py, might take awhile...")
            # process = subprocess.Popen(['MMPBSA.py', '-y', traj,
            #                        '-i', 'mmpbsa_input.txt',
            #                        '-cp', complex_prmtop,
            #                        '-rp', 'nosapo.prmtop',
            #                        '-lp', 'noslig.prmtop'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #
            subprocess.run(['MMPBSA.py', '-y', traj,
                                   '-i', 'mmpbsa_input.txt',
                                   '-cp', complex_prmtop,
                                   '-rp', 'nosapo.prmtop',
                                   '-lp', 'noslig.prmtop'], capture_output=True, check=True)
            #
            # decomp = self.decomp_to_csv('FINAL_DECOMP_MMPBSA.dat', 'decomp.csv')
            results = results_to_csv('FINAL_RESULTS_MMPBSA.dat', 'result.csv')
            logger.log(results.iloc[-1])


def decomp_to_csv(decomp_filename, csv_filename):
    with open(decomp_filename, 'r') as f:
        header = ''
        while 'Resid 1' not in header:
            header = f.readline()
        h1 = header.strip().split(',')
        h2 = f.readline()
        h2 = h2.strip().split(',')
        resid = h1[:2]
        h1, h2 = list(dict.fromkeys(h1[2:])), list(dict.fromkeys(h2))
        h1.remove('')
        h2.remove('')
        header = ','.join(resid + list(map(' '.join, list(itertools.product(h1, h2))))) + '\n'
        with open(csv_filename, 'w') as csvfile:
            csvfile.write(header)
            csvfile.writelines(f.readlines())
    return pd.read_csv(csv_filename)


def results_to_csv(results_filename, csv_filename):
    with open(results_filename, 'r') as f:
        header = ''
        while 'Differences (Complex - Receptor - Ligand)' not in header:
            header = f.readline()
        header = f.readline()
        header = [h.strip() for h in header.strip().split('  ') if h]
        header = ','.join(header) + '\n'
        with open(csv_filename, 'w') as csvfile:
            csvfile.write(header)
            f.readline()
            while True:
                l = f.readline()
                if '---' not in l:
                    l = [c.strip() for c in l.split('   ') if c]
                    l = ','.join(l) + '\n'
                    csvfile.write(l)
                if 'TOTAL' in l:
                    break
    return pd.read_csv(csv_filename)


def prepare_mcmc(topology, config):
    sequence_move = LangevinSplittingDynamicsMove(
        timestep=config.parameters.integrator_params['timestep'],
        n_steps=config.n_steps,
        collision_rate=config.parameters.integrator_params['collision_rate'],
        reassign_velocities=True,
        n_restart_attempts=6,
        constraint_tolerance=config.parameters.integrator_setConstraintTolerance)

    if config.hybrid:
        atoms = md.Topology.from_openmm(topology).select("resn UNL")
        rot_move = MCRotationMove(atom_subset=atoms)
        trans_move = MCDisplacementMove(atom_subset=atoms,displacement_sigma=config.displacement_sigma * unit.angstrom)
        sequence_move = SequenceMove([trans_move, rot_move, sequence_move])

    return sequence_move


class SystemParams(Config):
    def __init__(self, config_dict):
        # noinspection PyUnresolvedReferences
        import simtk.openmm as mm
        self.createSystem = None
        self.platform = None
        self.minMaxIters = None
        self.integrator_setConstraintTolerance = None
        self.platform_config = None
        self.integrator_params = None
        for k, v in config_dict.items():
            if k != "platform_config" and isinstance(v, dict):
                for k_, v_ in v.items():
                    try:
                        exec('v[k_] = ' + v_)
                    except TypeError:
                        v[k_] = v_
            else:
                exec('config_dict[k] = ' + str(v))
        self.__dict__.update(config_dict)
