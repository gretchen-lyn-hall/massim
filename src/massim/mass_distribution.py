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
    """Pick masses from a list of empirical data,
    
    Input can be a Pandas DataFrame, or a csv file. By default, masses are
    selected from a column named `mass`, but this can be overridden by
    setting 'key_column'.

    If the data contains frequency information, specify the column with
    `prob_column`. All frequency data will be normalized to sum to 1. If
    no frequency column is specified, all masses will have equal change of
    being selected.

    Masses are drawn from the list *without* replacement, so this class
    can never draw more masses than were initially specified.
    """
    def __init__(self, masses, mass_min=150, mass_max=1400, key_column='mass',
                 prob_column=None):
        self.mass_min = mass_min
        self.mass_max = mass_max
        prob_vals = None

        if isinstance(masses, str):
            if masses.endswith(".csv"):
                data = pd.read_csv(masses, index_col=False)
                if not key_column in data.columns:
                    raise KeyError("Mass .csv file has no mass column named "
                                   f"'{key_column}'.")
                m_vals = data[key_column].values
                if prob_column:
                    if not prob_column in data.columns:
                        raise KeyError("Mass .csv file has no probability column named "
                                       f"'{prob_column}'.")
                    prob_vals = data[prob_column].values
            else:
                raise ValueError("Input file must be .csv")
        elif isinstance(masses, pd.DataFrame):
            if not key_column in masses.columns:
                raise KeyError("Mass .csv file has no mass column named "
                               f"'{key_column}'.")
            m_vals = masses[key_column].values
            if prob_column:
                if not prob_column in masses.columns:
                    raise KeyError("Mass .csv file has no probability column named "
                                   f"'{prob_column}'.")
                prob_vals = masses[prob_column].values
            
        elif isinstance(masses, Iterable):
            m_vals = np.array(masses)
        else:
            raise ValueError("'masses' must be filename or list of masses")

        valid = ((m_vals >= mass_min) & (m_vals <= mass_max))
        self.masses = m_vals[valid]
        if prob_vals is not None:
            prob_vals = prob_vals[valid]
            prob_vals = prob_vals / prob_vals.sum()
        self.prob_vals = prob_vals

    def __call__(self, N: int, rng:np.random.Generator) -> np.ndarray:
        return rng.choice(self.masses, replace=False, size=N, p=self.prob_vals)

        
