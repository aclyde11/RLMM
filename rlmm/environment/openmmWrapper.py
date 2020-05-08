from io import StringIO

import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.utils.config import Config


class SystemParams(Config):
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if k != "platform_config" and isinstance(v, dict):
                for k_, v_ in v.items():
                    exec('v[k_] = ' + v_)
            else:
                exec('config_dict[k] = ' + str(v))
        self.__dict__.update(config_dict)
        print(config_dict)


class OpenMMSimulationWrapper:
    class Config(Config):
        def __init__(self, args):
            self.parameters = SystemParams(args['params'])
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

        def get_obj(self, system_loader, ln=None):
            self.systemloader = system_loader
            return OpenMMSimulationWrapper(self, ln)

    def __init__(self, config_: Config, ln=None):
        """

        :param systemLoader:
        :param config:
        """
        self.config = config_
        if ln is None:
            system = self.config.systemloader.get_system(self.config.parameters.createSystem)
        else:
            system = ln.system

        integrator = self.config.parameters.integrator(*self.config.parameters.integrator_params.values())

        integrator.setConstraintTolerance(self.config.parameters.integrator_setConstraintTolerance)

        # prepare simulation
        print(self.config.parameters.platform_config)
        self.simulation = app.Simulation(self.config.systemloader.get_topology(), system, integrator,
                                         self.config.parameters.platform, self.config.parameters.platform_config)
        self.simulation.context.setPositions(self.config.systemloader.get_positions())

        # minimize
        self.simulation.minimizeEnergy()

        # equilibrate for 100 steps
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

