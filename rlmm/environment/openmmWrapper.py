from io import StringIO

import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.utils.config import Config


class SystemParams(Config):

    config_modules = {'mm': mm, 'app': app, 'unit': unit}

    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    v[k_] = SystemParams.parse(v_)
            else:
                config_dict[k] = SystemParams.parse(str(v))
        self.__dict__.update(config_dict)

    @staticmethod
    def parse(config_string):
        if SystemParams._hasMath(config_string):
            math_exp = []
            L = config_string.split()
            for s in L:
                if SystemParams._isNumber(s) or SystemParams._hasMath(s):
                    math_exp.append(s)
                elif s.split('.')[0] in SystemParams.config_modules:
                    math_exp.append(s)
                else:
                    raise NameError(f'module: {s}, not found!')
            return eval(''.join(math_exp))
        else:
            if config_string.strip().split('.')[0] in SystemParams.config_modules:
                return eval(config_string)
            else:
                raise NameError(f'module: {config_string}, not found!')

    # @staticmethod
    # def parse(config_string):
    #     L = config_string.strip().split('.')
    #     if SystemParams._hasMath(config_string):
    #         math_exp = []
    #         for s in L:
    #             if SystemParams._isNumber(s) or SystemParams._hasMath(s):
    #                 math_exp.append(s)
    #             elif s.split('.')[0] in SystemParams.config_modules:
    #                 math_exp.append("SystemParams._parseObj(SystemParams.config_modules[s.split('.')[0]], s.split('.')[1:])")
    #         try:
    #             return eval(''.join(math_exp))
    #         except TypeError:
    #             print(''.join(math_exp))
    #     else:
    #         return SystemParams._parseObj(SystemParams.config_modules[L[0]], L[1:])

    @staticmethod
    def _isNumber(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def _hasMath(x):
        return bool({'/','+','-','*','**'} & set(x))

    # @staticmethod
    # def _parseObj(mod, attr):
    #     if not attr:
    #         return mod
    #     else:
    #         try:
    #             return SystemParams._parseObj(getattr(mod, attr[0]), attr[1:])
    #         except AttributeError:
    #             raise


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
