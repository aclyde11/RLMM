from gym import spaces

from rlmm.utils.config import Config


class EuclidanActionSpace:
    class Config(Config):
        def __init__(self, configs):
            self.ligand_only = configs['ligand_only']
            self.minimize = configs['minimize']

        def get_obj(self):
            return EuclidanActionSpace(self)

    def __init__(self, config: Config):
        self.config = config

    def translate(self, pos, vel, x, y, z, ligand_only=None, minimize=False):
        """

        :param x:
        :param y:
        :param z:
        :param minimize:
        """
        if ligand_only is None:
            pos += np.array([x, y, z]) * unit.angstrom
        else:
            pos[ligand_only] += np.array([x, y, z]) * unit.angstrom

        # if minimize:
        #     self.simulation.minimizeEnergy()
        #     self.simulation.context.setVelocitiesToTemperature(self.config.parameters.integrator_params['temperature'])
        return pos, vel
    def rotate(self, pos, vel, around_x=0, around_y=0, around_z=0, ligand_only=None, minimize=False):
        if ligand_only is None:
            ligand_only = slice(len(pos))
        center = np.mean(pos[ligand_only], axis=0)
        pos[ligand_only] -= center
        
        a,b,c = around_z, around_y, around_x
        sin,cos = np.sin, np.cos
        yaw = np.array([[cos(a), -sin(a), 0],[sin(a), cos(a), 0], [0,0,1]])
        pitch = np.array([[cos(b), 0, sin(b)], [0,1,0], [-sin(b), 0, cos(b)]])
        roll = np.array([[1,0,0], [0, cos(c), -sin(c)], [0, sin(c), cos(c)]])
        rot_matrix = yaw.dot(pitch).dot(roll)
        
        pos[ligand_only] = rot_matrix.dot(pos[ligand_only].T).T
        pos[ligand_only] += center
        
        return pos, vel

    def get_gym_space(self):
        return spaces.Discrete(2)
