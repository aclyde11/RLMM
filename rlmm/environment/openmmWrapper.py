from io import StringIO

import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm import app

from rlmm.environment.pdbutils import AmberSystemLoader
from rlmm.utils.config import Config, Configurable

class SystemParams(Config):
    def __init__(self):
        super().__init__()
        params = {'createSystem': {
            'implicitSolvent': app.GBn2,
            'nonbondedMethod': app.CutoffNonPeriodic,
            'nonbondedCutoff': 2.0 * unit.nanometer,
            'constraints': app.HBonds
        },
        'integrator': mm.LangevinIntegrator,
        'integrator_params': {
            'temperature': 300 * unit.kelvin,
            'frictionCoeff': 1.0 / unit.picoseconds,
            'stepSize': 2.0 * unit.femtoseconds
        },
        'integrator_setConstraintTolerance': 0.00001,
        'platform': mm.Platform.getPlatformByName('CPU')}
        [setattr(self, k,v) for k,v in filter(lambda t: "_" not in t[0][0], params.items())]


class OpenMMSimulationWrapper(Configurable):
    class Config(Config):
        def __init__(self, args=None):
            self.parameters = SystemParams()
            self.systemloader = None
            if args is not None:
                self.__dict__.update(args)

    def __init__(self, config_: Config):
        """

        :param systemLoader:
        :param config:
        """
        super().__init__(config_)

        system = self.systemloader.get_system(self.parameters.createSystem)

        integrator = self.parameters.integrator(*self.parameters.integrator_params.values())

        integrator.setConstraintTolerance(self.parameters.integrator_setConstraintTolerance)

        # prepare simulation
        self.simulation = app.Simulation(self.systemloader.get_topology(), system, integrator, self.parameters.platform)
        self.simulation.context.setPositions(self.systemloader.get_positions())

        # minimize
        self.simulation.minimizeEnergy()

        # equilibrate for 100 steps
        self.simulation.context.setVelocitiesToTemperature(self.parameters.integrator_params['temperature'])

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
            self.simulation.context.setVelocitiesToTemperature(self.parameters.integrator_params['temperature'])

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
