from abc import abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd

from .distributions import Distribution, NormalDistribution, force_range

class MassDistribution:
    """ Base class for method of randomly selecting masses """
    def __init__(self, mass_min=150, mass_max=1400):
        self.mass_min = mass_min
        self.mass_max = mass_max

    @abstractmethod
    def __call__(self, N: int, rng:np.random.Generator) -> np.ndarray:
        pass

class RandomMasses(MassDistribution):
    def __init__(self, dist: Distribution|None=None,
                 mass_min=150, mass_max=1400):
        if dist is None:
            dist = NormalDistribution(600, 250)
        self.dist = dist
        self.mass_min = mass_min
        self.mass_max = mass_max

    def __call__(self, N: int, rng:np.random.Generator) -> np.ndarray:
        return force_range(self.dist, N, self.mass_min, self.mass_max,
                           rng=rng)
        
    
class MassListPicker:
    """Pick masses from a list"""
    def __init__(self, masses, mass_min=150, mass_max=1400, key_column='mass'):
        self.mass_min = mass_min
        self.mass_max = mass_max

        if isinstance(masses, str):
            if masses.endswith(".csv"):
                data = pd.read_csv(masses, index_col=False)
                if not key_column in data.columns:
                    raise KeyError("Mass .csv file has no mass column named "
                                   f"'{key_column}'.")
                m_vals = data[key_column].values
            else:
                raise ValueError("Input file must be .csv")
        elif isinstance(masses, pd.DataFrame):
            if not key_column in masses.columns:
                raise KeyError("Mass .csv file has no mass column named "
                               f"'{key_column}'.")
            m_vals = masses[key_column].values
            
        elif isinstance(masses, Iterable):
            m_vals = np.array(masses)
        else:
            raise ValueError("'masses' must be filename or list of masses")

        self.masses = m_vals[(m_vals >= mass_min) &
                             (m_vals <= mass_max)]

    def __call__(self, N: int, rng:np.random.Generator) -> np.ndarray:
        return rng.choice(self.masses, replace=False, size=N)

        
