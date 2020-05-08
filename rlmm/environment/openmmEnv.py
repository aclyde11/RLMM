import random

import gym
import numpy as np
from gym import spaces
from rlmm.utils.config import Config

from rdkit import Chem
from rdkit.Chem import AllChem
from openeye import oedocking, oechem

class OpenMMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    class Config(Config):
        def __init__(self, configs):
            self.__dict__.update(configs)

    def __init__(self, config_: Config):
        """

        :param systemloader:
        """
        gym.Env.__init__(self)
        self.config = config_

        self.systemloader = self.config.systemloader.get_obj()

        self.obs_processor = self.config.obsmethods.get_obj()

        self.action = self.config.actions.get_obj()
        self.action_space = self.action.get_gym_space()
        self.observation_space = self.setup_observation_space()
        self.out_number = 0
        self.reset()

    def setup_action_space(self):
        """

        :return:
        """
        return spaces.Discrete(2)

    def setup_observation_space(self):
        """

        :return:
        """
        return spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8)

    def get_obs(self):
        """

        :return:
        """
        coords = self.openmm_simulation.get_coordinates()
        out = self.obs_processor(coords)
        return out

    def getscores(self,actions, gsmis, prot, num_returns = 10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions)-1
        idxs = list(np.random.choice(len(actions), min(num_returns,len(actions) - 1), replace=False).flatten())

        protein = oechem.OEMol(prot)
        receptor = oechem.OEGraphMol()
        pos = np.mean(np.array(self.openmm_simulation.get_coordinates()[-20:], dtype=np.float32), axis=0) * 10
        print(pos)
        oedocking.OEMakeReceptor(receptor, protein, float(pos[0]), float(pos[1]), float(pos[2]), True)
        dockobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
        dockobj.Initialize(receptor)

        scores = []
        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = self.action.get_aligned_action(actions[idx], gsmis[idx])
                dockedpose = oechem.OEMol()
                dockobj.DockMultiConformerMolecule(dockedpose, new_mol)
                ds = dockedpose.GetEnergy()
                ps = dockobj.ScoreLigand(new_mol)
                print("SCORE",ds ,ps )
                if return_docked_pose:
                    new_mol = oechem.OEMol(dockedpose)
                    new_mol2 = oechem.OEMol(dockedpose)
                data.append((new_mol, new_mol2, gs, action))
                scores.append(ds)
            except:
                continue
        order = np.argsort(scores)
        data = [data[i] for i in order]
        return data

    def step(self, action, sim_steps=10):
        """

        :param action:
        :return:
        """
        self.openmm_simulation.get_pdb("test.pdb")
        pdb = oechem.OEMol()
        prot = oechem.OEMol()
        lig = oechem.OEMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        ifs = oechem.oemolistream("test.pdb")
        oechem.OEReadMolecule(ifs, pdb)
        ifs.close()
        if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
            print("crap")
            exit()

        self.action.update_mol_aligneer(lig)
        actions, gsmis = self.action.get_new_action_set()
        data = self.getscores(actions, gsmis, prot, return_docked_pose=False, num_returns=-1)
        not_worked=True
        idxs = list(range(len(data)))
        idx = idxs.pop(0)
        counter = 0
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = data[idx]
                self.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader, ln=self.systemloader)
                not_worked=False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)


        self.action.apply_action(new_mol2, action)

        self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
        self.out_number += 1
        for i in range(60):
            self.openmm_simulation.run(416) #4166
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
            self.out_number += 1

        obs = self.get_obs()

        return obs, \
               random.random(), \
               random.random(), \
               {}

    def reset(self):
        """

        :return:
        """
        self.action.setup(self.config.systemloader.ligand_file_name)
        self.openmm_simulation = self.config.openmmWrapper.get_obj(self.systemloader)
        self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
        self.out_number += 1
        for i in range(90):
            self.openmm_simulation.run(277) #2777
            self.openmm_simulation.get_pdb("rlmmtest/out_{}.pdb".format(self.out_number))
            self.out_number += 1

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
