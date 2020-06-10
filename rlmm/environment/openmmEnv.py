import os
import pickle

import gym
import numpy as np
from gym import spaces
from pymbar import timeseries
from simtk import unit

from rlmm.utils.config import Config
from rlmm.utils.loggers import make_message_writer
from rlmm.environment.openmmWrappers.utils import detect_ligand_flyaway, get_pocket_residues


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
            self.tempdir = None
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
            os.mkdir(f"{self.config.tempdir()}/movie")
            self.data = {'mmgbsa': [],
                         'dscores': [0],
                         'pscores': [0],
                         'iscores': [0],
                         'hscores': [0],
                         'actions': [self.systemloader.inital_ligand_smiles],
                         'times' : [],
                         'movie_nbforce' : [],
                         'movie_time' : [],
                         'movie_mmgbsa' : []
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

    def step(self, action, sim_steps=10):
        self.data['actions'].append(action)

        with self.logger("step") as logger:
            init_obs = self.get_obs()

            self.openmm_simulation.run(self.samples_per_step, self.sim_steps)
            self.openmm_simulation.run_amber_mmgbsa()
            traj = self.openmm_simulation.writetraj()
            flew_away, d = detect_ligand_flyaway(traj, self.pocket_residues, return_difference=True)
            logger.log(f"FLEWAWAY: {flew_away}, with distance {d}")

        return self.get_obs(), 0, False, {'flew_away': flew_away, 'init_obs': init_obs}

    def reset(self):
        """

        :return:
        """
        from tqdm import tqdm

        with self.logger("reset") as logger:
            self.config.tempdir.start_step(0)
            self.sim_time = 0 * unit.nanosecond
            self.action.setup(self.config.systemloader.ligand_file_name)
            self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)

            init_obs = self.get_obs()
            self.openmm_simulation.run(self.samples_per_step, self.sim_steps)
            self.openmm_simulation.run_amber_mmgbsa()
            traj = self.openmm_simulation.writetraj()
            self.pocket_residues = get_pocket_residues(traj)
            flew_away, d = detect_ligand_flyaway(traj, self.pocket_residues, return_difference=True)
            logger.log(f"FLEWAWAY: {flew_away}, with distance {d}")

        return self.get_obs(), 0, False, {'flew_away' : flew_away, 'init_obs' : init_obs}

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
