from .experiment import Stage, PipelineData, StageData, StageParameter, dist_parser
from .distributions import (DISTRIBUTIONS,
                            Distribution,
                            UniformDistribution,
                            LogRandomDistribution,
                            NormalDistribution)

from ._core import TransformList, TandemTransformer, gen_dist, PickMassMode
from ._core import RNG as cRNG

import numpy as np
import pandas as pd

def _npify(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, pd.Series):
        return arr.values
    return np.array(arr)
    

def make_transform_list(mass_delta: np.ndarray,
                        xfrm_weights: np.ndarray,
                        xfrm_mean_len: np.ndarray,
                        same_len: bool=False,
                        always_center=True):

    return TransformList(_npify(mass_delta),
                         _npify(xfrm_weights),
                         _npify(xfrm_mean_len),
                         _npify(xfrm_mean_len),
                         same_len,
                         always_center)
    
class TransformStage(Stage):
    REQUIRES = ["abundance", "species_info"]
    
    target_masses = StageParameter(int, 1000)
    intensity_scale = StageParameter(Distribution, UniformDistribution(0.5, 0.9),
                                     msg_parser=dist_parser)
    
    def default_name(self):
        return "apply_transforms"
    
    def __init__(self,
                 transforms: TransformList,
                 mass_min: float = 150,
                 mass_max: float = 1200,
                 min_intensity: float = 1e4,
                 thresh_ppm: float = 1.0,
                 pick_mass: PickMassMode = PickMassMode.PICK_BY_MASS,
                 mass_center: float = 600.0,
                 mass_scale: float = 50.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.min_intensity = min_intensity
        self.thresh_ppm = thresh_ppm
        self.pick_mass = pick_mass
        self.mass_center = mass_center
        self.mass_scale = mass_scale

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        intensity_scale = state.intensity_scale.to_cdist()
        mass_dist = gen_dist("normal", [525, 190])
        
        # Seed our c++ RNG from our python RNG.
        c_rng = cRNG(rng.integers(2**30))
        transformer = TandemTransformer(input.abundance,
                                        input.species_info["mass"].values,
                                        self.transforms,
                                        intensity_scale,
                                        mass_dist,
                                        self.mass_min,
                                        self.mass_max,
                                        self.min_intensity,
                                        self.thresh_ppm,
                                        self.pick_mass,
                                        self.mass_center,
                                        self.mass_scale
                                        )
        
        xresult = transformer.apply_transforms(c_rng,
                                               target_masses=state.target_masses)
        species_info = input.species_info.iloc[xresult.original_ids()]
        species_info.loc[:,"mass"] = xresult.masses()
        base_response = xresult.extend_matrix(input.base_response)
        return PipelineData(
            input.copy(abundance=xresult.intensity(),
                       base_response = base_response,
                       species_info = species_info),
            messages=[], rng=rng)
                                        
                                        
                                        
        


def fit_logbeta(data):
    from types import SimpleNamespace
    result = {}
    import scipy.stats as ST
    data = np.log10(data)
    min_val = data.min()
    scale = data.max() - min_val
    data = (data - min_val) / scale
    fit = ST.fit(ST.beta, data, [(0, 20), (0, 20)])
    return SimpleNamespace(min=min_val, scale=scale, a=fit.params.a, b=fit.params.b)
