import copy
import math
import tempfile

import mdtraj as md
import numpy as np
import openmmtools
from openmmtools import cache
from openmmtools.multistate import ReplicaExchangeSampler
from simtk import unit
from tqdm import tqdm

import rlmm.environment.openmmWrappers.utils as mmWrapperUtils
from rlmm.utils.config import Config
from rlmm.utils.loggers import DCDReporter, make_message_writer


class MCMCReplicaExchangeOpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.t_max_k = None
            self.t_min_k = None
            self.n_replicas = None
            self.temps_in_k = None
            self.hybrid = None
            self.ligand_pertubation_samples = None
            self.displacement_sigma = None
            self.verbose = None
            self.n_steps = None
            self.parameters = mmWrapperUtils.SystemParams(args['params'])
            self.warmupparameters = None
            if "warmupparams" in args:
                self.warmupparameters = mmWrapperUtils.SystemParams(args['warmupparams'])

            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

            if self.temps_in_k is None:
                self.T_min = self.t_min_k * unit.kelvin  # Minimum temperature.
                self.T_max = self.t_max_k * unit.kelvin  # Maximum temperature.
                self.temps_in_k = [
                    self.T_min + (self.T_max - self.T_min) * (math.exp(float(i) / float(self.n_replicas - 1)) - 1.0) / (
                            math.e - 1.0) for i in range(self.n_replicas)]
                print("MCMCReplicaExchange Temps", self.temps_in_k)
            elif None in [self.T_min, self.T_max, self.n_replicas]:
                self.temps_in_k = self.temps_in_k * unit.kelvin
                self.T_min = min(self.temps_in_k)
                self.T_max = max(self.temps_in_k)
                self.n_replicas = len(self.temps_in_k)
            else:
                assert (False)

        def get_obj(self, system_loader, *args, **kwargs):
            self.systemloader = system_loader
            return MCMCReplicaExchangeOpenMMSimulationWrapper(self, *args, **kwargs)

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
                self.system = self.config.systemloader.get_system(self.config.parameters.createSystem)
                cache.global_context_cache.set_platform(self.config.parameters.platform,
                                                        self.config.parameters.platform_config)
                cache.global_context_cache.time_to_live = 10

            else:
                self.system = self.config.systemloader.system
            self.topology = self.config.systemloader.topology
            positions, velocities = self.config.systemloader.get_positions(), None


            sequence_move = mmWrapperUtils.prepare_mcmc(self.topology, self.config)

            protocol = {'temperature': self.config.temps_in_k}
            if self.explicit:
                protocol['pressure'] =  [1.0 * unit.atmosphere ] * self.config.n_replicas

            thermo_states = [openmmtools.states.ThermodynamicState(copy.deepcopy(self.system), temperature=t, pressure=1.0 * unit.atmosphere if self.config.systemloader.explicit else None) for t in self.config.temps_in_k]
            sampler_states = [openmmtools.states.SamplerState(positions=positions, velocities=velocities,
                                                                   box_vectors=self.config.systemloader.boxvec) for _ in
                                   thermo_states]

            self.simulation = ReplicaExchangeSampler(mcmc_moves=sequence_move, number_of_iterations=500)
            self.storage_path = tempfile.NamedTemporaryFile(delete=True).name + '.nc'
            self.reporter = openmmtools.multistate.MultiStateReporter(self.storage_path, checkpoint_interval=10)
            self.simulation.create(thermodynamic_states=thermo_states,
                                   sampler_states=sampler_states,
                                   storage=self.reporter)

            logger.log("Minimizing...", self.simulation.Status)
            self.simulation.minimize(max_iterations=self.config.parameters.minMaxIters)
            self.simulation.equilibrate(1)
            logger.log("Done, minimizing...", self.simulation.Status)

            # for idx in range(self.config.n_replicas):
            #     ctx = cache.global_context_cache.get_context(self.simulation._thermodynamic_states[idx])[0]
            #     ctx.setPositions(self.simulation.sampler_states[idx].positions)
            #     ctx.setVelocitiesToTemperature(self.config.temps_in_k[idx])
            #     self.simulation.sampler_states[idx].velocities = ctx.getState(getVelocities=True).getVelocities()


    def run(self, iters, steps_per_iter, idx=0):
        """

        :param steps:
        """
        with self.logger("run") as logger:

            if 'cur_sim_steps' not in self.__dict__:
                self.cur_sim_steps = 0.0 * unit.picosecond

            pbar = tqdm(range(1), desc="running {} steps per sample".format(steps_per_iter))
            self._trajs = np.zeros((iters, self.system.getNumParticles(), 3))
            self._times = np.zeros((iters))
            dcdreporter = DCDReporter(f"{self.config.tempdir()}/traj.dcd", 1, append=False)
            for i in pbar:
                self.simulation.run(1)
                self.cur_sim_steps += (steps_per_iter * self.get_sim_time())

                positions = self.simulation.sampler_states[idx].positions
                boxvectors = self.simulation.sampler_states[idx].box_vectors

                dcdreporter.report_ns(self.topology, positions, boxvectors, (i + 1), 0.5 * unit.femtosecond)

                # log trajectory
                self._trajs[i] = np.array(positions.value_in_unit(unit.angstrom)).reshape(
                    (self.system.getNumParticles(), 3))
                self._times[i] = self.cur_sim_steps.value_in_unit(unit.picosecond)
            pbar.close()
            exit()

    def writetraj(self, idx=0):
        if self.explicit:
            lengths, angles = mmWrapperUtils.get_mdtraj_box(boxvec=self.simulation.sampler_states[idx].box_vectors,
                                                            iterset=self._trajs.shape[0])
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology),
                                 unitcell_lengths=lengths,
                                 unitcell_angles=angles, time=self._times)
            traj.image_molecules(inplace=True)
        else:
            traj = md.Trajectory(self._trajs, md.Topology.from_openmm(self.topology), time=self._times)

        traj.save_hdf5(f'{self.config.tempdir()}/mdtraj_traj.h5')

    def run_amber_mmgbsa(self, run_decomp=False):
        mmWrapperUtils.run_amber_mmgbsa(self.logger, self.explicit, self.config.tempdir(), run_decomp=run_decomp)

    def get_sim_time(self):
        return self.config.n_steps * self.config.parameters.integrator_params['timestep']

    def get_velocities(self, idx=0):
        return self.simulation.sampler_states[idx].velocities

    def get_coordinates(self, idx=0):
        return mmWrapperUtils.get_coordinates_samplers(self.topology, self.simulation.sampler_states[idx],
                                                       self.explicit)

    def get_pdb(self, file_name=None, idx=0):
        mmWrapperUtils.get_pdb(self.topology, self.get_coordinates(idx), file_name=file_name)
