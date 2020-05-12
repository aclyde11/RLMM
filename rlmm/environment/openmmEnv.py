import random

import gym
import numpy as np
from gym import spaces
from pymbar import timeseries
from simtk import unit
from simtk.openmm import app
import sys
from rlmm.utils.config import Config


def make_message_writer(verbose, class_name):
    class MessageWriter(object):
        def __init__(self, method_name):
            self.class_name = class_name
            self.verbose = verbose
            self.method_name = method_name

        def log(self, *args, **kwargs):
            if self.verbose:
                print("[{}:{}]".format(self.class_name, self.method_name), *args, **kwargs)

        def __enter__(self):
            self.log("Entering")
            return self

        def __exit__(self, *args, **kwargs):
            self.log("Exiting")

    return MessageWriter


class OpenMMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    class Config(Config):
        def __init__(self, configs):
            self.__dict__.update(configs)

    def __init__(self, config_: Config, sim_steps, samples_per_step=60, movie_frames=60, verbose=True):
        self.logger = make_message_writer(verbose, self.__class__.__name__)
        with self.logger("__init__"):
            gym.Env.__init__(self)
            self.config = config_
            self.sim_steps = sim_steps
            self.movie_sample = int(samples_per_step / movie_frames)
            self.systemloader = self.config.systemloader.get_obj()
            self.samples_per_step = samples_per_step
            self.obs_processor = self.config.obsmethods.get_obj()
            self.action = self.config.actions.get_obj()
            self.action_space = self.action.get_gym_space()
            self.observation_space = self.setup_observation_space()
            self.out_number = 0
            self.verbose = verbose

            self.data = {'mmgbsa': [],
                         'docking_scores': [0],
                         'pose_scores': [0],
                         'actions': [self.systemloader.config.ligand_smiles]
                         }
            self.reset()

    def setup_action_space(self):
        with self.logger("setup_action_space") as logger:
            pass
        return spaces.Discrete(2)

    def setup_observation_space(self):
        with self.logger("setup_observation_space") as logger:
            pass
        return spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8)

    def get_obs(self):
        with self.logger("setup_observation_space") as logger:
            coords = self.openmm_simulation.get_coordinates()
            out = self.obs_processor(coords)
        return out

    def subsample(self, enthalpies):
        """
        Subsamples the enthalpies using John Chodera's code.
        This is probably better than the simple cutoff we normally use.
        No output -- it modifies the lists directly
        """
        for phase in enthalpies:
            # Use automatic equilibration detection and pymbar.timeseries to subsample
            [t0, g, Neff_max] = timeseries.detectEquilibration(enthalpies[phase])
            enthalpies[phase] = enthalpies[phase][t0:]
            indices = timeseries.subsampleCorrelatedData(enthalpies[phase], g=g)
            enthalpies[phase] = enthalpies[phase][indices]

    def mmgbsa(self, enthalpies):
        """
        Returns DeltaG, errDeltaG : float
            Estimated free energy of binding
        """
        self.subsample(enthalpies)

        DeltaH = dict()
        varDeltaH = dict()
        errDeltaH = dict()
        # TODO: it calculates standard error rather than variance
        for phase in enthalpies:
            DeltaH[phase] = enthalpies[phase].mean()
            varDeltaH[phase] = enthalpies[phase].std() ** 2
            errDeltaH[phase] = varDeltaH[phase] / len(enthalpies[phase])

        # TODO: Shouldn't have to use names?
        try:
            DeltaH['diff'] = 2 * DeltaH['complex']
        except:
            DeltaH['diff'] = 2 * DeltaH['com']
        varDeltaH['diff'] = 0
        errDeltaH['diff'] = 0
        for phase in enthalpies:
            DeltaH['diff'] -= DeltaH[phase]
            varDeltaH['diff'] += varDeltaH[phase]
            errDeltaH['diff'] += errDeltaH[phase]

        errDeltaH['diff'] = np.sqrt(errDeltaH['diff'])
        return DeltaH['diff'], errDeltaH['diff']

    def step(self, action, sim_steps=10):
        from tqdm import tqdm
        self.data['actions'].append(action)

        with self.logger("step") as logger:
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
            self.out_number += 1

            enthalpies = {'apo': np.zeros((self.samples_per_step)),
                          'com': np.zeros((self.samples_per_step)),
                          'lig': np.zeros((self.samples_per_step))}
            for i in tqdm(range(self.samples_per_step), desc="running {} steps per sample".format(self.sim_steps)):
                enthalpies['apo'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={0,
                                                                                                                  2}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)
                enthalpies['com'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={0,
                                                                                                                  1}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)
                enthalpies['lig'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={3}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)

                self.openmm_simulation.run(self.sim_steps)
                if i % self.movie_sample == 0:
                    self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
                    self.out_number += 1
            mmgbsa, err = self.mmgbsa(enthalpies)
            self.data['mmgbsa'].append((mmgbsa, err))

            logger.log('dgbind', mmgbsa, err)
            obs = self.get_obs()

        return obs, \
               random.random(), \
               random.random(), \
               {'energies': enthalpies}

    def reset(self):
        """

        :return:
        """
        from tqdm import tqdm

        with self.logger("reset") as logger:
            self.action.setup(self.config.systemloader.ligand_file_name)
            self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))

            self.out_number += 1
            enthalpies = {'apo': np.zeros((self.samples_per_step)),
                          'com': np.zeros((self.samples_per_step)),
                          'lig': np.zeros((self.samples_per_step))}
            pbar = tqdm(range(self.samples_per_step), desc="running {} steps per sample".format(self.sim_steps * 2))
            for i in pbar:
                rmsd = np.mean(np.abs(self.openmm_simulation.simulation.context.getState(getForces=True, groups={4}).getForces(asNumpy=True)))
                pbar.set_postfix({"rmsd" : rmsd})
                enthalpies['apo'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={0,
                                                                                                                  2}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)
                enthalpies['com'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={0,
                                                                                                                  1}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)
                enthalpies['lig'][i] = self.openmm_simulation.simulation.context.getState(getEnergy=True, groups={3}).getPotentialEnergy().value_in_unit(
                    unit.kilojoule / unit.mole)
                self.openmm_simulation.run(self.sim_steps * 2)  # 2777
                if i % self.movie_sample == 0:
                    self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
                    self.out_number += 1
            pbar.close()
            mmgbsa, err = self.mmgbsa(enthalpies)
            self.data['mmgbsa'].append((mmgbsa, err))
            logger.log('dgbind', mmgbsa, err)
        return self.get_obs()

    def render(self, mode='human', close=False):
        """

        :param mode:
        :param close:
        """
        pass

    def close(self):
        """

        """
        pass
