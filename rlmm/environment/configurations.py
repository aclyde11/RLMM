from enum import Enum, auto

from rlmm.environment.obsmethods import CoordinatePCA, Voxelizer

class StateSpaces(Enum):
    EUCLIDIAN = auto()

    def describe(self):
        return self.name, self.value

    def __str__(self):
        return 'my custom str! {0}'.format(self.value)

    @classmethod
    def default(cls):
        return cls.EUCLIDIAN

class ObservationMethods(Enum):
    COORDINATE_PCA = CoordinatePCA
    VOXELIZER = Voxelizer

    def describe(self):
        return self.name, self.value

    def __str__(self):
        return 'my custom str! {0}'.format(self.value)

    @classmethod
    def default(cls):
        return cls.COORDINATE_PCA


