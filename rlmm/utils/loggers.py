import sys

def make_message_writer(verbose_, class_name_):
    class MessageWriter(object):
        class_name = class_name_

        def __init__(self, method_name, verbose=None, enter_message=True):
            if verbose is None:
                self.verbose = verbose_
            else:
                self.verbose = verbose
            self.method_name = method_name
            self.enter_message = enter_message

        def log(self, *args, **kwargs):
            if self.verbose:
                print("INFO [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs)

        def error(self, *args, **kwargs):
            print("ERROR [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)

        def failure(self, *args, exit_all=False, **kwargs):
            print("FAILURE [{}:{}]".format(self.class_name, self.method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        @classmethod
        def static_failure(cls, method_name, *args, exit_all=False, **kwargs):
            print("FAILURE [{}:{}]".format(cls.class_name, method_name), *args, **kwargs, file=sys.stderr)
            if exit_all:
                exit()

        def __enter__(self):
            if self.enter_message:
                self.log("Entering")
            return self

        def __exit__(self, *args, **kwargs):
            if self.enter_message:
                self.log("Exiting")

    return MessageWriter

import time
import math
from simtk import unit
from simtk import openmm as mm

from simtk.openmm.app import DCDFile
from simtk.unit import nanometer

class DCDReporter(object):
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.
    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        """Create a DCDReporter.
        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            mode = 'r+b'
        else:
            mode = 'wb'
        self._out = open(file, mode)
        self._dcd = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, topology, state, currentStep, stepsize):
        """Generate a report.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        if self._dcd is None:
            self._dcd = DCDFile(
                self._out, topology, stepsize,
                currentStep, self._reportInterval, self._append
            )
        self._dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

    def __del__(self):
        self._out.close()

class StateDataReporter(object):
    """StateDataReporter outputs information about a simulation, such as energy and temperature, to a file.

    To use it, create a StateDataReporter, then add it to the Simulation's list of reporters.  The set of
    data to write is configurable using boolean flags passed to the constructor.  By default the data is
    written in comma-separated-value (CSV) format, but you can specify a different separator to use.
    """

    def __init__(self, file, reportInterval, step=False, time=False, potentialEnergy=False, kineticEnergy=False, totalEnergy=False, temperature=False, volume=False, density=False,
                 progress=False, remainingTime=False, speed=False, elapsedTime=False, separator=',', systemMass=None, totalSteps=None):
        """Create a StateDataReporter.

        Parameters
        ----------
        file : string or file
            The file to write to, specified as a file name or file object
        reportInterval : int
            The interval (in time steps) at which to write frames
        step : bool=False
            Whether to write the current step index to the file
        time : bool=False
            Whether to write the current time to the file
        potentialEnergy : bool=False
            Whether to write the potential energy to the file
        kineticEnergy : bool=False
            Whether to write the kinetic energy to the file
        totalEnergy : bool=False
            Whether to write the total energy to the file
        temperature : bool=False
            Whether to write the instantaneous temperature to the file
        volume : bool=False
            Whether to write the periodic box volume to the file
        density : bool=False
            Whether to write the system density to the file
        progress : bool=False
            Whether to write current progress (percent completion) to the file.
            If this is True, you must also specify totalSteps.
        remainingTime : bool=False
            Whether to write an estimate of the remaining clock time until
            completion to the file.  If this is True, you must also specify
            totalSteps.
        speed : bool=False
            Whether to write an estimate of the simulation speed in ns/day to
            the file
        elapsedTime : bool=False
            Whether to write the elapsed time of the simulation in seconds to
            the file.
        separator : string=','
            The separator to use between columns in the file
        systemMass : mass=None
            The total mass to use for the system when reporting density.  If
            this is None (the default), the system mass is computed by summing
            the masses of all particles.  This parameter is useful when the
            particle masses do not reflect their actual physical mass, such as
            when some particles have had their masses set to 0 to immobilize
            them.
        totalSteps : int=None
            The total number of steps that will be included in the simulation.
            This is required if either progress or remainingTime is set to True,
            and defines how many steps will indicate 100% completion.
        """
        self._reportInterval = reportInterval
        self._openedFile = False
        if (progress or remainingTime) and totalSteps is None:
            raise ValueError('Reporting progress or remaining time requires total steps to be specified')

        self._out = file
        self._step = step
        self._time = time
        self._potentialEnergy = potentialEnergy
        self._kineticEnergy = kineticEnergy
        self._totalEnergy = totalEnergy
        self._temperature = temperature
        self._volume = volume
        self._density = density
        self._progress = progress
        self._remainingTime = remainingTime
        self._speed = speed
        self._elapsedTime = elapsedTime
        self._separator = separator
        self._totalMass = systemMass
        self._totalSteps = totalSteps
        self._hasInitialized = False
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needEnergy = potentialEnergy or kineticEnergy or totalEnergy or temperature


    def report(self, system, state, currentStep):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(system)
            headers = self._constructHeaders()
            print('#"%s"' % ('"'+self._separator+'"').join(headers), file=self._out)
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = currentStep
            self._hasInitialized = True

        # Query for the values
        values = self._constructReportValues(state, currentStep)

        # Write the values.
        print(self._separator.join(str(v) for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _constructReportValues(self, state, currentStep):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        """
        values = []
        box = state.getPeriodicBoxVectors()
        volume = box[0][0]*box[1][1]*box[2][2]
        clockTime = time.time()
        if self._progress:
            values.append('%.1f%%' % (100.0*currentStep/self._totalSteps))
        if self._step:
            values.append(currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(unit.picosecond))
        if self._potentialEnergy:
            values.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._kineticEnergy:
            values.append(state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._totalEnergy:
            values.append((state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole))
        if self._temperature:
            values.append((2*state.getKineticEnergy()/(self._dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin))
        if self._volume:
            values.append(volume.value_in_unit(unit.nanometer**3))
        if self._density:
            values.append((self._totalMass/volume).value_in_unit(unit.gram/unit.item/unit.milliliter))
        if self._speed:
            elapsedDays = (clockTime-self._initialClockTime)/86400.0
            elapsedNs = (state.getTime()-self._initialSimulationTime).value_in_unit(unit.nanosecond)
            if elapsedDays > 0.0:
                values.append('%.3g' % (elapsedNs/elapsedDays))
            else:
                values.append('--')
        if self._elapsedTime:
            values.append(time.time() - self._initialClockTime)
        if self._remainingTime:
            elapsedSeconds = clockTime-self._initialClockTime
            elapsedSteps = currentStep-self._initialSteps
            if elapsedSteps == 0:
                value = '--'
            else:
                estimatedTotalSeconds = (self._totalSteps-self._initialSteps)*elapsedSeconds/elapsedSteps
                remainingSeconds = int(estimatedTotalSeconds-elapsedSeconds)
                remainingDays = remainingSeconds//86400
                remainingSeconds -= remainingDays*86400
                remainingHours = remainingSeconds//3600
                remainingSeconds -= remainingHours*3600
                remainingMinutes = remainingSeconds//60
                remainingSeconds -= remainingMinutes*60
                if remainingDays > 0:
                    value = "%d:%d:%02d:%02d" % (remainingDays, remainingHours, remainingMinutes, remainingSeconds)
                elif remainingHours > 0:
                    value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
                elif remainingMinutes > 0:
                    value = "%d:%02d" % (remainingMinutes, remainingSeconds)
                else:
                    value = "0:%02d" % remainingSeconds
            values.append(value)
        return values

    def _initializeConstants(self, system):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (Simulation) The simulation to generate a report for
        """
        if self._temperature:
            # Compute the number of degrees of freedom.
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0*unit.dalton:
                    dof += 3
            for i in range(system.getNumConstraints()):
                p1, p2, distance = system.getConstraintParameters(i)
                if system.getParticleMass(p1) > 0*unit.dalton or system.getParticleMass(p2) > 0*unit.dalton:
                    dof -= 1
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3
            self._dof = dof


    def _constructHeaders(self):
        """Construct the headers for the CSV output

        Returns: a list of strings giving the title of each observable being reported on.
        """
        headers = []
        if self._progress:
            headers.append('Progress (%)')
        if self._step:
            headers.append('Step')
        if self._time:
            headers.append('Time (ps)')
        if self._potentialEnergy:
            headers.append('Potential Energy (kJ/mole)')
        if self._kineticEnergy:
            headers.append('Kinetic Energy (kJ/mole)')
        if self._totalEnergy:
            headers.append('Total Energy (kJ/mole)')
        if self._temperature:
            headers.append('Temperature (K)')
        if self._volume:
            headers.append('Box Volume (nm^3)')
        if self._density:
            headers.append('Density (g/mL)')
        if self._speed:
            headers.append('Speed (ns/day)')
        if self._elapsedTime:
            headers.append('Elapsed Time (s)')
        if self._remainingTime:
            headers.append('Time Remaining')
        return headers

    def __del__(self):
        if self._openedFile:
            self._out.close()