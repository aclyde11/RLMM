"""
RLMM
RLMM is a reinforcement learning env for molecular modeling (currently only protein-ligand docking).
"""

# Add imports here
from .rlmm import * #lgtm [py/polluting-import]
from .environment import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
