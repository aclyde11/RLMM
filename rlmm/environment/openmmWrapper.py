from io import StringIO

import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.utils.config import Config
from rlmm.utils.exceptions import BadConfigError


class SystemParams(Config):

    config_modules = {'mm': mm, 'app': app, 'unit': unit}

    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    v[k_] = self._parse(v_)
            else:
                config_dict[k] = self._parse(str(v))
        self.__dict__.update(config_dict)

    def _parse(self, config_string):
        """
        Parse the system load configurations detailed in the <configurations>.yaml.
            - Valid module, method, and object calls of modules: 'mm', 'app', 'unit'

        :param config_string: str
        :return: eval(conf_string): Object
        """
        try:
            self._safeEval(config_string, self.config_modules)
        except (NameError, SyntaxError):
            raise BadConfigError(f'Unrecognized option during config-file parsing: {config_string}')


    def _safeEval(self, s, local):
        return eval(s, {'__builtins__': {}}, local)



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

    def translate(self, x, y, z, minimize=True):
        """

        :param x:
        :param y:
        :param z:
        :param minimize:
        """
        pos = self.simulation.context.getState(getPositions=True, getVelocities=True)
        pos = pos.getPositions(asNumpy=True)
        # pos[5082:5125] += np.array([x, y, z]) * unit.angstrom

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

    def get_pdb(self):
        """

        :return:
        """
        output = StringIO()
        app.PDBFile.writeFile(self.simulation.topology,
                              self.simulation.context.getState(getPositions=True).getPositions(),
                              file=output)
        return output.getvalue()
