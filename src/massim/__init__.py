"""
MASSIM - Ecological simulation framework for synthesizing FT-ICR data.


"""
__author__ = "Gretchen Hall (gretchenhall@arizona.edu)"
from importlib.metadata import version
__version__ = version("massim")


from . import distributions
from . import experiment
from . import gradient_response
from . import species_config
from . import sampling
from . import profile
from . import noise
from . import filters
from . import mass_distribution
from . import stats
from . import transformations
from . import data_analysis

__all__ = ['distributions',
           'experiment',
           'gradient_response',
           'species_config',
           'sampling',
           'profile',
           'noise',
           'filters',
           'mass_distribution',
           'transformations',
           'stats',
           'data_analysis',
           ]


# from .distributions import *
# from .gradient_response import GradientResponse, BetaResponse
# from .mass_distribution import RandomMasses, MassListPicker
# from .species_config import (
#     SpeciesGroupConfig,
#     GenSpeciesStage,
#     GenCoreStage,
#     quick_compas)
# from .sampling import *
# from .experiment import (Experiment,
#                          Message,
#                          OutputFilter,
#                          ParameterSweep,
#                          LinSpace,
#                          ArraySweep,
#                          RandomSweep,
#                          ReplicateStage,
#                          DebugStage,
#                          )
# from .transformations import (TransformStage, make_transform_list)
# from .stats import *

