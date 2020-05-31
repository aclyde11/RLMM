import mdtraj as md
import numpy as np
from openmmtools import integrators
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm

import rlmm.environment.openmmWrappers.utils as mmWrapperUtils
from rlmm.utils.config import Config
from rlmm.utils.loggers import DCDReporter
from rlmm.utils.loggers import make_message_writer


class OpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.verbose = None
            self.parameters = mmWrapperUtils.SystemParams(args['params'])
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader, *args, **kwargs):
            self.systemloader = system_loader
            return OpenMMSimulationWrapper(self, *args, **kwargs)

    def __init__(self, config_: Config, old_sampler_state=None):
        """

        :param systemLoader:
        :param config:
        """
        self._times = None
        self.config = config_
        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        if self.config.systemloader.system is None:
            self.system = self.config.systemloader.get_system(self.config.parameters.createSystem)
            self.topology = self.config.systemloader.topology
            positions, velocities = self.config.systemloader.get_positions(), None

        else:
            self.system = self.config.systemloader.system
            self.topology = self.config.systemloader.topology
            positions, velocities = self.config.systemloader.get_positions(), None

        self.topology = self.config.systemloader.topology
        self.explicit = self.config.systemloader.explicit

        integrator = integrators.GeodesicBAOABIntegrator(temperature=self.config.parameters.integrator_params['temperature'],
                                                    timestep=self.config.parameters.integrator_params['timestep'],
                                                    collision_rate=self.config.parameters.integrator_params[
                                                        'collision_rate'],
                                                    constraint_tolerance=self.config.parameters.integrator_setConstraintTolerance)
        self.simulation = app.Simulation(self.topology, self.system, integrator,
                                         self.config.parameters.platform, self.config.parameters.platform_config)
        self.simulation.context.setPositions(self.config.systemloader.positions)
        self.simulation.minimizeEnergy(self.config.parameters.minMaxIters)
        self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])

    def get_sim_time(self):
        return self.config.parameters.integrator_params['timestep'] *self.config.n_steps

    def get_coordinates(self):
        return mmWrapperUtils.get_coordinates(self.topology,
                                              self.simulation.context.getState(getPositions=True).getPositions(),
                                              self.simulation.system.getDefaultPeriodicBoxVectors(),
                                              self.simulation.system.getNumParticles(),
                                              self.explicit)

    def get_velocities(self):
        return self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def run_amber_mmgbsa(self, run_decomp=False):
        return mmWrapperUtils.run_amber_mmgbsa(self.logger, self.explicit, self.config.tempdir(), run_decomp=run_decomp)

    def get_pdb(self, file_name=None):
        return mmWrapperUtils.get_pdb(self.simulation.topology,
                                      self.simulation.context.getState(getPositions=True).getPositions(),
                                      file_name=file_name)

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
                self.simulation.step(self.config.n_steps * steps_per_iter)
                self.cur_sim_steps += (steps_per_iter * self.get_sim_time())
                _state = self.simulation.context.getState(getPositions=True)
                dcdreporter.report(self.topology, _state, (i + 1), 0.5 * unit.femtosecond)

                # log trajectory
                self._trajs[i] = np.array(_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(
                    (self.system.getNumParticles(), 3))
                self._times[i] = self.cur_sim_steps.value_in_unit(unit.picosecond)
            pbar.close()

    def writetraj(self):
        if self.explicit:
            lengths, angles = mmWrapperUtils.get_mdtraj_box(
                boxvec=self.simulation.system.getDefaultPeriodicBoxVectors(),
                iterset=self._trajs.shape[0])
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=lengths,
                                 unitcell_angles=angles, time=self._times)
            traj.image_molecules(inplace=True)
        else:
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology), time=self._times)

        # traj.save_pdb(f'{self.config.tempdir()}/mdtraj_traj.pdb')
        traj.save_hdf5(f'{self.config.tempdir()}/mdtraj_traj.h5')
        # traj.save_dcd(f'{self.config.tempdir()}/mdtraj_traj.dcd')
        return traj