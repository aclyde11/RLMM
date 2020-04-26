from io import StringIO

import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.utils.config import Config


class SystemParams(Config):
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    exec('v[k_] = ' + v_)
            else:
                exec('config_dict[k] = ' + str(v))
        self.__dict__.update(config_dict)


class OpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.parameters = SystemParams(args['params'])
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader):
            self.systemloader = system_loader
            return OpenMMSimulationWrapper(self)

    def __init__(self, config_: Config):
        """

        :param systemLoader:
        :param config:
        """
        self.config = config_
        system = self.config.systemloader.get_system(self.config.parameters.createSystem)

        integrator = self.config.parameters.integrator(*self.config.parameters.integrator_params.values())

        integrator.setConstraintTolerance(self.config.parameters.integrator_setConstraintTolerance)

        # prepare simulation
        self.simulation = app.Simulation(self.config.systemloader.get_topology(), system, integrator,
                                         self.config.parameters.platform)
        self.simulation.context.setPositions(self.config.systemloader.get_positions())

        # minimize
        self.simulation.minimizeEnergy()

        # equilibrate for 100 steps
        self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])
    def apply_action_simulation(self, action, *args, **kwargs):
        pos = self.get_coordinates()
        vel = self.get_velocities()
        new_pos, new_vel = action(pos=pos, vel=vel, args, kwargs)
        self.set_coordinates(pos)
        self.set_velocities(vel)
        # simulation.translate(*action, ligand_only=self.config.ligand_only, minimize=self.config.minimize)

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

    def set_coordinates(self, new_coordinates):
        """

        :param new_coordinates:
        """
        self.simulation.context.setPositions(new_coordinates)
    def get_velocities(self):
        """

        :return:
        """
        return self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def set_velocities(self, new_velocities):
        """

        :param new_velocities:
        """
        self.simulation.context.setVelocities(new_velocities)

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

