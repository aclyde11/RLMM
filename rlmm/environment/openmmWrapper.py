import copy
import math
import sys
from glob import glob
from io import StringIO

import mdtraj as md
import mdtraj.utils as mdtrajutils
import numpy as np
import simtk.openmm as mm
from openmmtools import cache
from openmmtools import integrators
from openmmtools import multistate
from openmmtools.mcmc import WeightedMove, MCMCSampler, LangevinSplittingDynamicsMove, SequenceMove, \
    MCDisplacementMove, MCRotationMove, HMCMove, GHMCMove
from openmmtools.states import ThermodynamicState, SamplerState
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm
import subprocess
import tempfile
from rlmm.utils.config import Config
from rlmm.utils.loggers import StateDataReporter, DCDReporter
from rlmm.utils.loggers import make_message_writer


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


class MCMCReplicaOpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.tempdir = None
            self.hybrid = None
            self.ligand_pertubation_samples = None
            self.displacement_sigma = None
            self.verbose = None
            self.n_steps = None
            self.n_replicas = 4
            self.T_min = 275.0 * unit.kelvin
            self.T_max = 370.0 * unit.kelvin
            self.parameters = SystemParams(args['params'])
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader, *args, **kwargs):
            self.systemloader = system_loader
            return MCMCReplicaOpenMMSimulationWrapper(self, *args, **kwargs)

    def __init__(self, config_: Config, old_sampler_state=None):
        """

        :param systemLoader:
        :param config:
        """
        self.config = config_
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__") as logger:
            self.explicit = self.config.systemloader.explicit
            if self.config.systemloader.system is None:
                system = self.config.systemloader.get_system(self.config.parameters.createSystem)
                self.topology = self.config.systemloader.topology
                cache.global_context_cache.set_platform(self.config.parameters.platform,
                                                        self.config.parameters.platform_config)
                cache.global_context_cache.time_to_live = 10
                prot_atoms = None
            else:
                system = self.config.systemloader.system
                self.topology = self.config.systemloader.topology
                past_sampler_state_velocities = [old_sampler_state.simulation.sampler_states[i].velocities for i in
                                                 range(self.config.n_replicas)]
                prot_atoms = md.Topology.from_openmm(self.topology).select("protein")

            temperatures = [self.config.parameters.integrator_params['temperature']] + [
                self.config.T_min + (self.config.T_max - self.config.T_min) * (
                        math.exp(float(i) / float((self.config.n_replicas - 1) - 1)) - 1.0) / (math.e - 1.0) for i
                in range(self.config.n_replicas - 1)]
            logger.log("Running with replica temperatures", temperatures)
            self.thermodynamic_states = [ThermodynamicState(system=system, temperature=T,
                                                            pressure=1.0 * unit.atmosphere if self.config.systemloader.explicit else None)
                                         for T in temperatures]

            atoms = md.Topology.from_openmm(self.topology).select("resn UNK or resn UNL")
            logger.log("ligand atom positions", atoms)
            subset_move = MCDisplacementMove(atom_subset=atoms,
                                             displacement_sigma=self.config.displacement_sigma * unit.angstrom)
            subset_rot = MCRotationMove(atom_subset=atoms)
            ghmc_move = HMCMove(timestep=self.config.parameters.integrator_params['timestep'],
                                n_steps=self.config.n_steps)

            langevin_move = LangevinSplittingDynamicsMove(
                timestep=self.config.parameters.integrator_params['timestep'],
                n_steps=self.config.n_steps,
                collision_rate=self.config.parameters.integrator_params['collision_rate'],
                reassign_velocities=False,
                n_restart_attempts=6,
                constraint_tolerance=self.config.parameters.integrator_setConstraintTolerance)

            if self.config.hybrid:
                langevin_move_weighted = WeightedMove([(ghmc_move, 0.5),
                                                       (langevin_move, 0.5)])
                sequence_move = SequenceMove([subset_move, subset_rot, langevin_move_weighted])
            else:
                sequence_move = SequenceMove([subset_move, subset_rot, langevin_move])

            self.simulation = multistate.MultiStateSampler(mcmc_moves=sequence_move, number_of_iterations=np.inf)
            files = glob(self.config.tempdir + 'multistate_*.nc')
            storage_path = self.config.tempdir + 'multistate_{}.nc'.format(len(files))
            self.reporter = multistate.MultiStateReporter(storage_path, checkpoint_interval=1)
            self.simulation.create(thermodynamic_states=self.thermodynamic_states, sampler_states=[
                SamplerState(self.config.systemloader.get_positions(), box_vectors=self.config.systemloader.boxvec) for
                i in range(self.config.n_replicas)], storage=self.reporter)

            self.simulation.minimize(max_iterations=self.config.parameters.minMaxIters)

            if prot_atoms is not None:
                for replica in range(self.config.n_replicas):
                    velocities = self.simulation.sampler_states[replica].velocities
                    for prot_atom in prot_atoms:
                        velocities[prot_atom] = past_sampler_state_velocities[replica][prot_atom]
                    self.simulation.sampler_states[replica].velocities = velocities

    def run(self, steps):
        """

        :param steps:
        """
        self.simulation.run(steps)

    def get_sim_time(self):
        return self.config.n_steps * self.config.parameters.integrator_params['timestep'] * self.config.n_replicas

    def get_velocities(self, index=0):
        """

        :return:
        """
        return None

    def get_coordinates(self, index=0):
        """

        :return:
        """
        if self.explicit:
            pos = self.simulation.sampler_states[index].positions
            trajectory_positions = np.array(unit.Quantity(pos, pos[0].unit).value_in_unit(unit.angstrom))
            trajectory_positions = trajectory_positions.reshape(
                (1, trajectory_positions.shape[0], trajectory_positions.shape[1]))
            a, b, c = self.simulation.sampler_states[index].box_vectors
            a, b, c = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom), c.value_in_unit(unit.angstrom)
            a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(a, b, c)
            trajectory_box_lengths = [[a, b, c]]
            trajectory_box_angles = [[alpha, beta, gamma]]

            traj = md.Trajectory(trajectory_positions, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)
            traj = traj.image_molecules(inplace=False)
            coords = traj.xyz.reshape((traj.n_atoms, 3))
        else:
            coords = self.simulation.sampler_states[index].positions
        return coords

    def get_pdb(self, file_name=None, index=None):
        """

        :return:
        """
        if file_name is None:
            output = StringIO()
        else:
            output = open(file_name, 'w')

        app.PDBFile.writeFile(self.topology,
                              self.get_coordinates() if index is None else self.get_coordinates(index),
                              file=output)
        if file_name is None:
            return output.getvalue()
        else:
            output.close()
            return True

    def get_enthalpies(self):
        with self.logger("get_enthalpies", enter_message=False) as logger:
            if not self.explicit:
                return 0, 0, 0
            trajectory_positions = self.simulation.sampler_states[0].positions

            if 'mmgbsa_contexts' not in self.__dict__:
                logger.log("building contexts")
                self.mmgbsa_contexts = {}
                self.mmgbsa_idx = {}

                traj = md.Topology.from_openmm(self.topology)

                seles = ["not (water or resn HOH or resn NA or resn CL)", "protein", "resn UNK or resn UNL"]
                seles = zip(["com", "apo", "lig"], seles)
                for phase, sele in seles:
                    idx = traj.select(sele)
                    self.mmgbsa_idx[phase] = idx
                    topology = traj.subset(idx).to_openmm()
                    system = self.config.systemloader.openmm_system_generator.create_system(topology)
                    logger.log(f"Built {phase} system")
                    system.setDefaultPeriodicBoxVectors(
                        *self.config.systemloader.modeller.getTopology().getPeriodicBoxVectors())
                    dummyIntegrator = mm.LangevinIntegrator(self.config.parameters.integrator_params['temperature'],
                                                            self.config.parameters.integrator_params['collision_rate'],
                                                            self.config.parameters.integrator_params['timestep'])
                    ctx = mm.Context(system, dummyIntegrator, mm.Platform.getPlatformByName('CPU'))
                    self.mmgbsa_contexts[phase] = ctx

            values = {}
            for phase in ['com', 'apo', 'lig']:
                self.mmgbsa_contexts[phase].setPositions(trajectory_positions[self.mmgbsa_idx[phase]])
                values[phase] = self.mmgbsa_contexts[phase].getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)

        return values['com'], values['apo'], values['lig']


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
            self._trajs = []
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

            thermodynamic_state = ThermodynamicState(system=system,
                                                     temperature=self.config.parameters.integrator_params[
                                                         'temperature'],
                                                     pressure=1.0 * unit.atmosphere if self.config.systemloader.explicit else None)

            sampler_state = SamplerState(positions=positions, velocities=velocities,
                                         box_vectors=self.config.systemloader.boxvec)

            atoms = md.Topology.from_openmm(self.topology).select("resn UNK or resn UNL")
            logger.log("ligand atom positions", atoms)
            subset_move = MCDisplacementMove(atom_subset=atoms,
                                             displacement_sigma=self.config.displacement_sigma * unit.angstrom)
            subset_rot = MCRotationMove(atom_subset=atoms)
            ghmc_move = GHMCMove(timestep=self.config.parameters.integrator_params['timestep'],
                                 n_steps=self.config.n_steps,
                                 collision_rate=self.config.parameters.integrator_params['collision_rate'])

            langevin_move = LangevinSplittingDynamicsMove(
                timestep=self.config.parameters.integrator_params['timestep'],
                n_steps=self.config.n_steps,
                collision_rate=self.config.parameters.integrator_params['collision_rate'],
                reassign_velocities=False,
                n_restart_attempts=6,
                constraint_tolerance=self.config.parameters.integrator_setConstraintTolerance)

            if self.config.hybrid:
                langevin_move_weighted = WeightedMove([(ghmc_move, 0.5),
                                                       (langevin_move, 0.5)])
                sequence_move = SequenceMove([subset_move, subset_rot, langevin_move_weighted])
            else:
                sequence_move = SequenceMove([subset_move, subset_rot, langevin_move])

            self.sampler = MCMCSampler(thermodynamic_state, sampler_state, move=sequence_move)
            self.sampler.minimize(max_iterations=self.config.parameters.minMaxIters)

            if prot_atoms is not None:
                velocities = self.sampler.sampler_state.velocities
                for prot_atom in prot_atoms:
                    velocities[prot_atom] = past_sampler_state_velocities[prot_atom]
                self.sampler.sampler_state.velocities = velocities
            self.setup_component_contexts()

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

    def get_mdtraj_box(self, boxvec=None, a=None, b=None, c=None):
        if boxvec is not None:
            a, b, c = boxvec
        elif None in [a, b, c]:
            self.logger.static_failure('get_mdtraj_box', f"a {a}, b {b}, c {c}, boxvec {boxvec} are all None.",
                                       exit_all=True)

        a, b, c = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom), c.value_in_unit(unit.angstrom)
        a, b, c = np.array(a), np.array(b), np.array(c)
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(a, b, c)
        return a, b, c, alpha, beta, gamma

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
            for i in pbar:
                self.sampler.run(steps_per_iter)
                self.cur_sim_steps += (steps_per_iter * self.get_sim_time())

                # log trajectory
                self._trajs[i] = np.array(self.sampler.sampler_state.positions.value_in_unit(unit.angstrom)).reshape(
                    (1, self.system.getNumParticles(), 3))
                self._times[i] = self.cur_sim_steps.value_in_unit(unit.picosecond)

            pbar.close()

    def run_amber_mmgbsa(self):
        import shutil, os, itertools
        from rlmm.environment.systemloader import working_directory

        os.mkdir(f"{self.config.tempdir}env_steps/{self._id_number}")

        for phase, ext in itertools.product(['apo', 'lig', 'us_com', 'com'], ['prmtop', 'inpcrd']):
            shutil.move(f"{self.config.tempdir}{phase}_{self._id_number}.{ext}", f"{self.config.tempdir}env_steps/{self._id_number}/{phase}_{self._id_number}.{ext}")

        with working_directory(f"{self.config.tempdir}env_steps/{self._id_number}"):
            traj = md.Trajectory(self._trajs, time=self._times, topology=md.Topology.from_openmm(self.topology))
            traj.image_molecules(inplace=True)
            traj.save_mdcrd("traj.mdcrd")

            args = ['MMPBSA.py',
                    '-i', 'input.txt',
                    '-lp', 'lig.prmtop',
                    '-rp', 'apo.prmtop',
                    '-cp', 'us_com.prmtop',
                    '-y', 'traj.mdcrd']
            res = subprocess.run(args)
            exit()




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
            trajectory_positions = np.array(unit.Quantity(pos, pos[0].unit).value_in_unit(unit.angstrom))
            trajectory_positions = trajectory_positions.reshape(
                (1, trajectory_positions.shape[0], trajectory_positions.shape[1]))
            a, b, c = self.sampler.sampler_state.box_vectors
            a, b, c = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom), c.value_in_unit(unit.angstrom)
            a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(a, b, c)
            trajectory_box_lengths = [[a, b, c]]
            trajectory_box_angles = [[alpha, beta, gamma]]

            traj = md.Trajectory(trajectory_positions, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

            traj = traj.image_molecules(inplace=False)
            coords = traj.xyz.reshape((traj.n_atoms, 3))
        else:
            coords = self.sampler.sampler_state.positions
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

    def get_nb_matrix(self, groups=None):
        with self.logger("get_nb_matrix", enter_message=False) as logger:
            if 'mmgbsa_contexts' not in self.__dict__:
                logger.failure("MMGBSA_CONTEXT NOT SET BUT GET_ENTHAPALIES CALLED", exit_all=True)

            ctx = cache.global_context_cache.get_context(self.sampler.thermodynamic_state)[0]
            forces = ctx.getState(getForces=True).getForces(asNumpy=True)

        return {'apo': forces[self.mmgbsa_idx['apo']], 'lig': forces[self.mmgbsa_idx['lig']]}

    def setup_component_contexts(self):
        with self.logger("setup_component_contexts", enter_message=True) as logger:
            if not self.explicit:
                return 0, 0, 0
            trajectory_positions = self.sampler.sampler_state.positions

            if 'mmgbsa_contexts' not in self.__dict__:
                logger.log("building contexts")
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
                    logger.log(f"Built {phase} system")
                    system.setDefaultPeriodicBoxVectors(
                        *self.config.systemloader.modeller.getTopology().getPeriodicBoxVectors())
                    dummyIntegrator = mm.LangevinIntegrator(self.config.parameters.integrator_params['temperature'],
                                                            self.config.parameters.integrator_params['collision_rate'],
                                                            self.config.parameters.integrator_params['timestep'])
                    ctx = mm.Context(system, dummyIntegrator, mm.Platform.getPlatformByName('CPU'))
                    self.mmgbsa_contexts[phase] = ctx

    def get_mmgbsa(self):
        com, lig, apo = self.get_enthalpies()
        return com - lig - apo

    def get_enthalpies(self, groups=None):
        with self.logger("get_enthalpies", enter_message=False) as logger:
            if 'mmgbsa_contexts' not in self.__dict__:
                logger.error("MMGBSA_CONTEXT NOT SET BUT GET_ENTHAPALIES CALLED")
                return 0, 0, 0
            if not self.explicit:
                return 0, 0, 0
            trajectory_positions = self.sampler.sampler_state.positions

            values = {}
            for phase in ['com', 'apo', 'lig']:
                self.mmgbsa_contexts[phase].setPositions(trajectory_positions[self.mmgbsa_idx[phase]])
                values[phase] = self.mmgbsa_contexts[phase].getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)

        return values['com'], values['apo'], values['lig']


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
