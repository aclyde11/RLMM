import os
import pickle

import gym
import numpy as np
from gym import spaces
from pymbar import timeseries
from simtk import unit

from rlmm.utils.config import Config
from rlmm.utils.loggers import make_message_writer


class EnvStepData:

    def __init__(self):
        self.topology = None
        self.md_traj_obj = None
        self.simulation_start_time = None
        self.simulation_end_time = None
        self.mmgbsa = None


class EpisodeData:
    def __init__(self):
        self.steps = []

    def log_trah(self, traj : EnvStepData ):
        self.steps.append(traj)


class OpenMMEnvLogger:
    def __init__(self):
        self.config = None
        self.episodes = []

    def log_episode_data(self, ep : EpisodeData):
        self.episodes.append(ep)

    def save_checkpoint(self):
        pass

    @staticmethod
    def load_from_checkpoint():
        pass


class OpenMMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    class Config(Config):
        def __init__(self, configs):
            self.openmmWrapper = None
            self.actions = None
            self.obsmethods = None
            self.systemloader = None
            self.__dict__.update(configs)

    def __init__(self, config_: Config, ):
        self.config = config_

        self.logger = make_message_writer(self.config.verbose, self.__class__.__name__)
        with self.logger("__init__"):
            gym.Env.__init__(self)
            self.sim_steps = self.config.sim_steps
            self.movie_sample = int(self.config.samples_per_step / self.config.movie_frames)
            self.systemloader = self.config.systemloader.get_obj()
            self.samples_per_step = self.config.samples_per_step
            self.obs_processor = self.config.obsmethods.get_obj()
            self.action = self.config.actions.get_obj()
            self.action_space = self.action.get_gym_space()
            self.observation_space = self.setup_observation_space()
            self.out_number = 0
            self.verbose = self.config.verbose
            os.mkdir(self.config.tempdir + "movie")
            self.data = {'mmgbsa': [],
                         'dscores': [0],
                         'pscores': [0],
                         'iscores': [0],
                         'hscores': [0],
                         'actions': [self.systemloader.inital_ligand_smiles]
                         }

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
            out = self.obs_processor(self.openmm_simulation)
        return out

    def subsample(self, enthalpies):
        """
        Subsamples the enthalpies using John Chodera's code.
        This is probably better than the simple cutoff we normally use.
        No output -- it modifies the lists directly
        """
        for phase in enthalpies:
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
        for phase in enthalpies:
            DeltaH[phase] = enthalpies[phase].mean()
            varDeltaH[phase] = enthalpies[phase].std() ** 2
            errDeltaH[phase] = varDeltaH[phase] / len(enthalpies[phase])
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
            self.openmm_simulation.get_pdb(self.config.tempdir + "movie/out_{}.pdb".format(self.out_number))
            self.out_number += 1

            enthalpies = {'apo': np.zeros((self.samples_per_step)),
                          'com': np.zeros((self.samples_per_step)),
                          'lig': np.zeros((self.samples_per_step))}
            pbar = tqdm(range(self.samples_per_step), desc="running {} steps per sample".format(self.sim_steps))
            for i in pbar:
                self.openmm_simulation.run(self.sim_steps)
                if self.config.systemloader.explicit:
                    enthalpies['com'][i], enthalpies['apo'][i], enthalpies['lig'][i] = self.openmm_simulation.get_enthalpies()
                    pbar.set_postfix({"mmgbsa" : enthalpies['com'][i] - enthalpies['apo'][i] -  enthalpies['lig'][i]})
                if i % self.movie_sample == 0:
                    self.openmm_simulation.get_pdb(self.config.tempdir + "movie/out_{}.pdb".format(self.out_number))
                    self.out_number += 1
            pbar.close()
            if self.config.systemloader.explicit:
                mmgbsa, err = self.mmgbsa(enthalpies)
                self.data['mmgbsa'].append((mmgbsa, err))
                logger.log('mmgbsa', mmgbsa, err)

            else:
                mmgbsa, err = 0, 0
            obs = self.get_obs()

        return obs, \
               mmgbsa, \
               False, \
               {'energies' : enthalpies}

    def reset(self):
        """

        :return:
        """
        from tqdm import tqdm

        with self.logger("reset") as logger:
            self.action.setup(self.config.systemloader.ligand_file_name)
            self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)


            if self.config.equilibrate:
                samples_per_step = self.openmm_simulation.get_sim_time() * self.sim_steps
                steps = int(10 * unit.nanosecond / samples_per_step)
                logger.log(f"Equilbrate is set to True, running {steps} instead of {self.samples_per_step}")
            else:
                steps = self.samples_per_step
            ms = int(steps / self.config.movie_frames)

            self.openmm_simulation.get_pdb(self.config.tempdir + "movie/out_{}.pdb".format(self.out_number))

            self.out_number += 1
            enthalpies = {'apo': np.zeros((steps)),
                          'com': np.zeros((steps)),
                          'lig': np.zeros((steps))}
            pbar = tqdm(range(steps), desc="running {} steps per sample".format(self.sim_steps))
            for i in pbar:
                self.openmm_simulation.run(self.sim_steps)
                if self.config.systemloader.explicit:
                    enthalpies['com'][i], enthalpies['apo'][i], enthalpies['lig'][i] = self.openmm_simulation.get_enthalpies()
                    pbar.set_postfix({"mmgbsa" : enthalpies['com'][i] - enthalpies['apo'][i] -  enthalpies['lig'][i]})
                if i % ms == 0:
                    self.openmm_simulation.get_pdb(self.config.tempdir + "movie/out_{}.pdb".format(self.out_number))
                    self.out_number += 1
            pbar.close()
            if self.config.systemloader.explicit:
                logger.log('mmgbsa', self.mmgbsa(enthalpies))
            with open('nb.pkl', 'wb') as f:
                pickle.dump(enthalpies, f)
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
