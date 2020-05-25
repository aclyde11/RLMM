import itertools
import itertools
import subprocess
import sys
from io import StringIO

import mdtraj as md
import mdtraj.utils as mdtrajutils
import numpy as np
import simtk.openmm as mm
from openmmtools import cache
from openmmtools import integrators
from openmmtools.mcmc import WeightedMove, MCMCSampler, LangevinSplittingDynamicsMove, SequenceMove, \
    MCDisplacementMove, MCRotationMove, GHMCMove
from openmmtools.states import ThermodynamicState, SamplerState
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm

from rlmm.utils.config import Config
from rlmm.utils.loggers import StateDataReporter, DCDReporter
from rlmm.utils.loggers import make_message_writer
from rlmm.environment.systemloader import working_directory


class SystemParams(Config):
    def __init__(self, config_dict):
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


class MCMCOpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.hybrid = None
            self.ligand_pertubation_samples = None
            self.displacement_sigma = None
            self.verbose = None
            self.n_steps = None
            self.parameters = SystemParams(args['params'])
            self.warmupparameters = None
            if "warmupparams" in args:
                self.warmupparameters = SystemParams(args['warmupparams'])

            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader, *args, **kwargs):
            self.systemloader = system_loader
            return MCMCOpenMMSimulationWrapper(self, *args, **kwargs)

    def __init__(self, config_: Config, old_sampler_state=None):
        """

        :param systemLoader:
        :param config:
        """
        self._times = None
        self.config = config_
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.explicit = self.config.systemloader.explicit
            self.amber = bool(self.config.systemloader.config.method == 'amber')
            self._trajs = np.zeros((1, 1))
            self._id_number = int(self.config.systemloader.params_written)

            if self.config.systemloader.system is None:
                system = self.config.systemloader.get_system(self.config.parameters.createSystem)
                self.system = system
                self.topology = self.config.systemloader.topology

                cache.global_context_cache.set_platform(self.config.parameters.platform,
                                                        self.config.parameters.platform_config)
                cache.global_context_cache.time_to_live = 10
                prot_atoms = None

                # positions, velocities = self.warmup(
                #     self.config.systemloader.get_warmup_system(self.config.warmupparameters.createSystem))
                # positions, velocities = self.relax_ligand((copy.deepcopy(system), self.topology, positions, velocities))
                # positions, velocities = self.relax((system, self.topology, positions, velocities))
                positions, velocities = self.config.systemloader.get_positions(), None

            else:
                system = self.config.systemloader.system
                self.system = system
                self.topology = self.config.systemloader.topology
                past_sampler_state_velocities = old_sampler_state.sampler.sampler_state.velocities
                prot_atoms = md.Topology.from_openmm(self.topology).select("protein")
                positions, velocities = self.config.systemloader.get_positions(), None

            langevin_move = LangevinSplittingDynamicsMove(
                timestep=self.config.parameters.integrator_params['timestep'],
                n_steps=self.config.n_steps,
                collision_rate=self.config.parameters.integrator_params['collision_rate'],
                reassign_velocities=False,
                n_restart_attempts=6,
                constraint_tolerance=self.config.parameters.integrator_setConstraintTolerance)

            if self.config.hybrid:
                atoms = md.Topology.from_openmm(self.topology).select("resn UNK or resn UNL")
                subset_pertub = WeightedMove([(MCRotationMove(atom_subset=atoms), 0.2),
                                              (MCDisplacementMove(atom_subset=atoms,
                                                                  displacement_sigma=self.config.displacement_sigma * unit.angstrom),
                                               0.8)])

                ghmc_move = GHMCMove(timestep=self.config.parameters.integrator_params['timestep'],
                                     n_steps=self.config.n_steps,
                                     collision_rate=self.config.parameters.integrator_params['collision_rate'])
                sequence_move = WeightedMove([(SequenceMove([subset_pertub, ghmc_move]), 0.5),
                                              (langevin_move, 0.5)])
            else:
                sequence_move = langevin_move

            self.sampler = MCMCSampler(ThermodynamicState(system=system,
                                                          temperature=self.config.parameters.integrator_params[
                                                              'temperature'],
                                                          pressure=1.0 * unit.atmosphere if self.config.systemloader.explicit else None)
                                       , SamplerState(positions=positions, velocities=velocities,
                                                      box_vectors=self.config.systemloader.boxvec), move=sequence_move)
            self.sampler.minimize(max_iterations=self.config.parameters.minMaxIters)

            # reassign protein velocities from prior simulation
            if prot_atoms is not None:
                velocities = self.sampler.sampler_state.velocities
                for prot_atom in prot_atoms:
                    velocities[prot_atom] = past_sampler_state_velocities[prot_atom]
                self.sampler.sampler_state.velocities = velocities

    def relax(self, system):
        with self.logger('relax') as logger:
            system, topology, positions, velocities = system

            integrator = integrators.GeodesicBAOABIntegrator(
                temperature=self.config.warmupparameters.integrator_params['temperature'],
                collision_rate=self.config.warmupparameters.integrator_params['collision_rate'],
                timestep=self.config.warmupparameters.integrator_params['timestep'],
                constraint_tolerance=self.config.warmupparameters.integrator_setConstraintTolerance)
            thermo_state = ThermodynamicState(system=system,
                                              temperature=self.config.warmupparameters.integrator_params['temperature'])

            context_cache = cache.ContextCache(self.config.warmupparameters.platform,
                                               self.config.warmupparameters.platform_config)
            context, context_integrator = context_cache.get_context(thermo_state,
                                                                    integrator)
            context.reinitialize(preserveState=True)

            context.setPositions(positions)
            context.setVelocities(velocities)
            context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

            mm.LocalEnergyMinimizer.minimize(context)
            velocities = context.getState(getVelocities=True).getVelocities()
            positions = context.getState(getPositions=True).getPositions()
            step_size = 100000
            updates = 100
            delta = int(step_size / updates)
            reporter = StateDataReporter(sys.stdout, 1, step=True, time=True, potentialEnergy=True,
                                         kineticEnergy=True, totalEnergy=True, temperature=True,
                                         progress=True, remainingTime=True, speed=True, elapsedTime=True,
                                         separator='\t',
                                         totalSteps=step_size * 100)
            dcdreporter = DCDReporter('relax.dcd', 1, append=False)

            _trajectory = np.zeros((updates, self.system.getNumParticles(), 3))
            for j in range(updates):
                context_integrator.step(delta)
                _ctx, _integrator = context_cache.get_context(thermo_state)
                _state = _ctx.getState(getPositions=True, getVelocities=True, getForces=True,
                                       getEnergy=True, getParameters=True, enforcePeriodicBox=False)
                system = _ctx.getSystem()
                positions, velocities = _state.getPositions(), _state.getVelocities()
                reporter.report(system, _state, delta * (j + 1))
                dcdreporter.report(topology, _state, delta * (j + 1), 0.5 * unit.femtosecond)
                _trajectory[j] = _state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

            a, b, c, alpha, beta, gamma = self.get_mdtraj_box(boxvec=self.config.systemloader.boxvec)
            _trajectory = md.Trajectory(_trajectory, md.Topology.from_openmm(topology),
                                        unitcell_lengths=np.array([[a, b, c] * _trajectory.shape[0]]).reshape(
                                            (_trajectory.shape[0], 3)),
                                        unitcell_angles=np.array([[alpha, beta, gamma] * _trajectory.shape[0]]).reshape(
                                            (_trajectory.shape[0], 3)))
            _trajectory.save_mdcrd("relax.crd")
            _trajectory.image_molecules(inplace=True)
            _trajectory.save_hdf5("relax.h5")
            _trajectory.save_mdcrd("relax_image.crd")

        return positions, velocities

    def get_mdtraj_box(self, boxvec=None, a=None, b=None, c=None, iterset=-1):
        if boxvec is not None:
            a, b, c = boxvec
        elif None in [a, b, c]:
            self.logger.static_failure('get_mdtraj_box', f"a {a}, b {b}, c {c}, boxvec {boxvec} are all None.",
                                       exit_all=True)

        a, b, c = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom), c.value_in_unit(unit.angstrom)
        a, b, c = np.array(a), np.array(b), np.array(c)
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(a, b, c)
        if iterset is -1:
            return a, b, c, alpha, beta, gamma
        else:
            return [[a,b,c]] * iterset, [[alpha, beta, gamma]] * iterset

    def relax_ligand(self, system):
        '''
        Performs equilibration step by restraining protein positions.
        :param system: MM system to use
        :return: positions and velocities after relaxation
        '''
        system, topology, positions, velocities = system

        ## BACKBONE RESTRAIN
        force = mm.CustomExternalForce('k_restr*periodicdistance(x, y, z, x0, y0, z0)^2')
        # Add the restraint weight as a global parameter in kcal/mol/A^2
        force.addGlobalParameter("k_restr", 5.0)
        # force.addGlobalParameter("k_restr", weight*unit.kilocalories_per_mole/unit.angstroms**2)
        # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        positions_ = positions.value_in_unit(unit.nanometer)
        for i, atom_id in enumerate(md.Topology.from_openmm(topology).select("backbone")):
            pos = positions_[atom_id]
            pops = mm.Vec3(pos[0], pos[1], pos[2])
            _ = force.addParticle(int(atom_id), pops)

        system.addForce(force)

        integrator = integrators.GeodesicBAOABIntegrator(
            temperature=self.config.warmupparameters.integrator_params['temperature'],
            collision_rate=self.config.warmupparameters.integrator_params['collision_rate'],
            timestep=self.config.warmupparameters.integrator_params['timestep'],
            constraint_tolerance=self.config.warmupparameters.integrator_setConstraintTolerance)
        thermo_state = ThermodynamicState(system=system,
                                          temperature=self.config.warmupparameters.integrator_params['temperature'])

        context_cache = cache.ContextCache(self.config.warmupparameters.platform,
                                           self.config.warmupparameters.platform_config)
        context, context_integrator = context_cache.get_context(thermo_state,
                                                                integrator)

        context.reinitialize(preserveState=True)
        context.setPositions(positions)
        context.setVelocities(velocities)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        mm.LocalEnergyMinimizer.minimize(context)
        velocities = context.getState(getVelocities=True).getVelocities()
        positions = context.getState(getPositions=True).getPositions()
        step_size = 100000
        updates = 100
        delta = int(step_size / updates)
        reporter = StateDataReporter(sys.stdout, 1, step=True, time=True, potentialEnergy=True,
                                     kineticEnergy=True, totalEnergy=True, temperature=True,
                                     progress=True, remainingTime=True, speed=True, elapsedTime=True, separator='\t',
                                     totalSteps=updates)

        _trajectory = np.zeros((updates, self.system.getNumParticles(), 3))
        for j in range(updates):
            context_integrator.step(delta)
            _ctx, _integrator = context_cache.get_context(thermo_state)
            _state = _ctx.getState(getPositions=True, getVelocities=True, getForces=True,
                                   getEnergy=True, getParameters=True, enforcePeriodicBox=False)
            positions, velocities = _state.getPositions(), _state.getVelocities()
            reporter.report(_ctx.getSystem(), _state, j)
            _trajectory[j] = _state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        a, b, c, alpha, beta, gamma = self.get_mdtraj_box(boxvec=self.config.systemloader.boxvec)
        _trajectory = md.Trajectory(_trajectory, md.Topology.from_openmm(topology),
                                    unitcell_lengths=np.array([[a, b, c] * _trajectory.shape[0]]).reshape(
                                        (_trajectory.shape[0], 3)),
                                    unitcell_angles=np.array([[alpha, beta, gamma] * _trajectory.shape[0]]).reshape(
                                        (_trajectory.shape[0], 3)))
        _trajectory.image_molecules(inplace=True)
        _trajectory.save_hdf5("relax_ligand.h5")
        return positions, velocities

    def warmup(self, system):
        from rlmm.utils.loggers import StateDataReporter
        system, topology, positions = system

        temperatures = [250 * unit.kelvin, 275 * unit.kelvin, 290 * unit.kelvin, 300 * unit.kelvin,
                        self.config.warmupparameters.integrator_params['temperature']]

        integrator = integrators.GeodesicBAOABIntegrator(temperature=temperatures[0],
                                                         collision_rate=self.config.warmupparameters.integrator_params[
                                                             'collision_rate'],
                                                         timestep=self.config.warmupparameters.integrator_params[
                                                             'timestep'],
                                                         constraint_tolerance=self.config.warmupparameters.integrator_setConstraintTolerance)
        thermo_state = ThermodynamicState(system=system, temperature=temperatures[0])

        context_cache = cache.ContextCache(self.config.warmupparameters.platform,
                                           self.config.warmupparameters.platform_config)
        context, context_integrator = context_cache.get_context(thermo_state,
                                                                integrator)
        context.setPositions(positions)
        mm.LocalEnergyMinimizer.minimize(context)
        context.setVelocitiesToTemperature(temperatures[0])
        velocities = context.getState(getVelocities=True)
        positions = context.getState(getPositions=True)

        step_size = 100000
        updates = 100
        delta = int(step_size / updates)
        reporter = StateDataReporter(sys.stdout, 1, step=True, time=True, potentialEnergy=False,
                                     kineticEnergy=False, totalEnergy=True, temperature=True, volume=False,
                                     density=False,
                                     progress=True, remainingTime=True, speed=True, elapsedTime=True, separator='\t',
                                     totalSteps=updates * len(temperatures),
                                     systemMass=np.sum([self.system.getParticleMass(pid) for pid in
                                                        range(self.system.getNumParticles())]))

        _trajectory = np.zeros((len(temperatures) * updates, self.system.getNumParticles(), 3))
        for i, temp in enumerate(temperatures):
            if i != 0:
                thermo_state = ThermodynamicState(system=system,
                                                  temperature=temp)
                integrator = integrators.GeodesicBAOABIntegrator(temperature=temp,
                                                                 collision_rate=
                                                                 self.config.warmupparameters.integrator_params[
                                                                     'collision_rate'],
                                                                 timestep=
                                                                 self.config.warmupparameters.integrator_params[
                                                                     'timestep'],
                                                                 constraint_tolerance=self.config.warmupparameters.integrator_setConstraintTolerance)
                context, context_integrator = context_cache.get_context(thermo_state,
                                                                        integrator)
                context.setPositions(positions)
                context.setVelocities(velocities)
                context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
            for j in range(updates):
                context_integrator.step(delta)
                _ctx, _ = context_cache.get_context(thermo_state)
                _state = _ctx.getState(getPositions=True, getVelocities=True, getForces=True,
                                       getEnergy=True, getParameters=True, enforcePeriodicBox=False)
                positions, velocities = _state.getPositions(), _state.getVelocities()
                reporter.report(_ctx.getSystem(), _state, i * j + j)
                _trajectory[i * updates + j] = _state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        a, b, c, alpha, beta, gamma = self.get_mdtraj_box(boxvec=self.config.systemloader.boxvec)
        _trajectory = md.Trajectory(_trajectory, md.Topology.from_openmm(topology),
                                    unitcell_lengths=np.array([[a, b, c] * _trajectory.shape[0]]).reshape(
                                        (_trajectory.shape[0], 3)),
                                    unitcell_angles=np.array([[alpha, beta, gamma] * _trajectory.shape[0]]).reshape(
                                        (_trajectory.shape[0], 3)))

        return positions, velocities

    def run(self, iters, steps_per_iter):
        """

        :param steps:
        """
        with self.logger("run") as logger:

            if 'cur_sim_steps' not in self.__dict__:
                self.cur_sim_steps = 0.0 * unit.picosecond

            pbar = tqdm(range(iters), desc="running {} steps per sample".format(steps_per_iter))
            self._trajs = np.zeros((iters, self.system.getNumParticles(), 3))
            self._times = np.zeros((iters))
            dcdreporter = DCDReporter(f"{self.config.tempdir()}/traj.dcd", 1, append=False)
            for i in pbar:
                self.sampler.run(steps_per_iter)
                self.cur_sim_steps += (steps_per_iter * self.get_sim_time())
                _state = cache.global_context_cache.get_context(self.sampler.thermodynamic_state)[0].getState(
                    getPositions=True)
                dcdreporter.report(self.topology, _state, (i + 1), 0.5 * unit.femtosecond)

                # log trajectory
                self._trajs[i] = np.array(self.sampler.sampler_state.positions.value_in_unit(unit.angstrom)).reshape(
                    (self.system.getNumParticles(), 3))
                self._times[i] = self.cur_sim_steps.value_in_unit(unit.picosecond)
            pbar.close()

    def decomp_to_csv(self, decomp_filename, csv_filename):
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

    def results_to_csv(self, results_filename, csv_filename):
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

    # def decomp_to_pandas(self, decomp_filename, csv_filename=None):
    #     if csv_filename:
    #         self.decomp_to_csv(decomp_filename, csv_filename)
    #         return pd.read_csv(csv_filename)
    #     else:
    #         self.decomp_to_csv(decomp_filename, '!tmp_decomp_to_pandas!.csv')
    #         res = pd.read_csv('!tmp_decomp_to_pandas!.csv')
    #         os.remove('!tmp_decomp_to_pandas!.csv')
    #         return res
    #
    # def results_to_pandas(results_filename, csv_filename=None):
    #     if csv_filename:
    #         results_to_csv(results_filename, csv_filename)
    #         return pd.read_csv(csv_filename)
    #     else:
    #         decomp_to_csv(results_filename, '!tmp_results_to_pandas!.csv')
    #         res = pd.read_csv('!tmp_results_to_pandas!.csv')
    #         os.remove('!tmp_results_to_pandas!.csv')
    #         return res

    def writetraj(self):
        if self.explicit:
            lengths, angles = self.get_mdtraj_box(boxvec=self.sampler.sampler_state.box_vectors, iterset=self._trajs.shape[0])
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=lengths,
                                 unitcell_angles=angles, time=self._times)
            traj.image_molecules(inplace=True)
        else:
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology), time=self._times)

        traj.save_pdb(f'{self.config.tempdir()}/mdtraj_traj.pdb')
        traj.save_hdf5(f'{self.config.tempdir()}/mdtraj_traj.h5')
        traj.save_dcd(f'{self.config.tempdir()}/mdtraj_traj.dcd')

    def run_amber_mmgbsa(self):
        with self.logger('run_amber_mmgbsa') as logger:
            complex_prmtop = f"com.prmtop"
            traj = "traj.dcd"
            with working_directory(self.config.tempdir()):
                if self.explicit:
                    with open("cpptraj_input.txt", 'w') as f:
                        f.write("strip :WAT parmout stripped.prmtop outprefix traj.dcd nobox\n" +
                                "trajout test2.dcd\n" +
                                "run\n")
                    proc = subprocess.run(['cpptraj', '-p', complex_prmtop, '-y', traj, '-i', 'cpptraj_input.txt'],
                                          check=True, capture_output=True)
                    complex_prmtop = "stripped.prmtop"
                    traj = "test2.dcd"

                proc = subprocess.run(['ante-MMPBSA.py',
                                       '-p', complex_prmtop,
                                       '-l', 'noslig.prmtop',
                                       '-r', 'nosapo.prmtop',
                                       '-n', ':UNL'], check=True, capture_output=True)

                with open("mmpbsa_input.txt", 'w') as f:
                    f.write(
                        '&general\nstartframe=1, endframe=100, interval=20,\nverbose=3, keep_files=1, strip_mask=":WAT:CL:CIO:CS:IB:K:LI:MG:NA:RB:HOH",\n/\n&gb\nigb=5, saltcon=0.150,\n/\n&decomp\nidecomp=3,csv_format=1\n/\n')

                logger.log("Running amber MMPBSA.py, might take awhile...")
                proc = subprocess.run(['MMPBSA.py', '-y', traj,
                                       '-i', 'mmpbsa_input.txt',
                                       '-cp', complex_prmtop,
                                       '-rp', 'nosapo.prmtop',
                                       '-lp', 'noslig.prmtop'], capture_output=True, check=True)

                self.decomp_to_csv('FINAL_DECOMP_MMPBSA.dat', 'decomp.csv')
                self.results_to_csv('FINAL_RESULTS_MMPBSA.dat', 'result.csv')

    def get_sim_time(self):
        return self.config.n_steps * self.config.parameters.integrator_params['timestep']

    def get_velocities(self):
        return self.sampler.sampler_state.velocities

    def get_coordinates(self):
        """

        :return:
        """
        if self.explicit:
            pos = self.sampler.sampler_state.positions
            trajectory_positions = np.array(pos.value_in_unit(unit.angstrom))
            trajectory_positions = trajectory_positions.reshape(
                (1, self.sampler.sampler_state.n_particles, 3))
            lengths, angles = self.get_mdtraj_box(boxvec=self.sampler.sampler_state.box_vectors, iterset=1)

            traj = md.Trajectory(trajectory_positions, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=lengths, unitcell_angles=angles)

            traj = traj.image_molecules(inplace=False)
            coords = traj.xyz.reshape((traj.n_atoms, 3))
        else:
            coords = self.sampler.sampler_state.positions.value_in_unit(unit.angstrom)
        return coords

    def get_pdb(self, file_name=None):
        """

        :return:
        """
        if file_name is None:
            output = StringIO()
        else:
            output = open(file_name, 'w')
        app.PDBFile.writeFile(self.topology,
                              self.get_coordinates(),
                              file=output, keepIds=False)
        if file_name is None:
            return output.getvalue()
        else:
            output.close()
            return True


class OpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.parameters = SystemParams(args['params'])
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader, *args, **kwargs):
            self.systemloader = system_loader
            return OpenMMSimulationWrapper(self, *args, **kwargs)

    def __init__(self, config_: Config, ln=None, prior_sim=None):
        """

        :param systemLoader:
        :param config:
        """
        self.config = config_
        if ln is None:
            system = self.config.systemloader.get_system(self.config.parameters.createSystem)
            prot_atoms = None
        else:
            system = self.config.systemloader.system
            prot_atoms = md.Topology.from_openmm(self.topology).select("protein")
        self.topology = self.config.systemloader.topology
        self.explicit = self.config.systemloader.explicit

        integrator = integrators.LangevinIntegrator(temperature=self.config.parameters.integrator_params['temperature'],
                                                    timestep=self.config.parameters.integrator_params['timestep'],
                                                    collision_rate=self.config.parameters.integrator_params[
                                                        'collision_rate'],
                                                    constraint_tolerance=self.config.parameters.integrator_setConstraintTolerance)

        # prepare simulation
        prior_sim_vel = None
        if prior_sim is not None:
            prior_sim_vel = prior_sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)
            del prior_sim
        self.simulation = app.Simulation(self.topology, system, integrator,
                                         self.config.parameters.platform, self.config.parameters.platform_config)
        self.simulation.context.setPositions(self.config.systemloader.positions)

        self.simulation.minimizeEnergy(self.config.parameters.minMaxIters)

        if prot_atoms is not None:
            self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])
            cur_vel = self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
            for i in prot_atoms:
                cur_vel[i] = prior_sim_vel[i]
            self.simulation.context.setVelocities(cur_vel)
        else:
            self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])

    def translate(self, x, y, z, ligand_only=None, minimize=True):
        """

        :param x:
        :param y:
        :param z:
        :param minimize:
        """
        pos = self.simulation.context.getState(getPositions=True, getVelocities=True)
        pos = pos.getPositions(asNumpy=True)

        if ligand_only is None:
            pos += np.array([x, y, z]) * unit.angstrom
        else:
            pos[ligand_only] += np.array([x, y, z]) * unit.angstrom

        if minimize:
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])

    def get_sim_time(self):
        return self.config.parameters.integrator_params['timestep']

    def run(self, steps):
        """

        :param steps:
        """
        self.simulation.step(steps)

    def get_coordinates(self):
        """

        :return:
        """
        return self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

    def get_pdb(self, file_name=None):
        """

        :return:
        """
        if file_name is None:
            output = StringIO()
        else:
            output = open(file_name, 'w')

        app.PDBFile.writeFile(self.simulation.topology,
                              self.simulation.context.getState(getPositions=True).getPositions(),
                              file=output)
        if file_name is None:
            return output.getvalue()
        else:
            output.close()
            return True

    def get_enthalpies(self, groups=None):
        if not self.explicit:
            return 0, 0, 0
        trajectory_positions = self.simulation.context.getState(getPositions=True).getPositions()

        if 'mmgbsa_contexts' not in self.__dict__:
            self.mmgbsa_contexts = {}
            self.mmgbsa_idx = {}

            traj = md.Topology.from_openmm(self.topology)

            seles = ["not (water or resn HOH or resn NA or resn CL)", "protein", "resn UNK or resn UNL"]
            seles = zip(["com", "apo", "lig"], seles)
            for phase, sele in seles:
                idx = traj.select(sele)
                self.mmgbsa_idx[phase] = idx
                pos = trajectory_positions[idx]
                topology = traj.subset(idx).to_openmm()
                system = self.config.systemloader.openmm_system_generator.create_system(topology)
                system.setDefaultPeriodicBoxVectors(
                    *self.config.systemloader.modeller.getTopology().getPeriodicBoxVectors())
                dummyIntegrator = mm.LangevinIntegrator(self.config.parameters.integrator_params['temperature'],
                                                        self.config.parameters.integrator_params['collision_rate'],
                                                        self.config.parameters.integrator_params['timestep'])
                ctx = mm.Context(system, dummyIntegrator, mm.Platform.getPlatformByName('CPU'))
                self.mmgbsa_contexts[phase] = ctx
                # logger.log("Minimizing")
                # mm.LocalEnergyMinimizer.minimize(ctx)

        values = {}
        for phase in ['com', 'apo', 'lig']:
            self.mmgbsa_contexts[phase].setPositions(trajectory_positions[self.mmgbsa_idx[phase]])
            values[phase] = self.mmgbsa_contexts[phase].getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole)

        return values['com'], values['apo'], values['lig']
