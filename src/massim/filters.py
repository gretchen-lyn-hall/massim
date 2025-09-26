from .experiment import Stage, PipelineData, StageData, StageParameter, dist_parser
from .distributions import (DISTRIBUTIONS,
                            Distribution,
                            UniformDistribution,
                            LogRandomDistribution,
                            NormalDistribution)


import numpy as np
import pandas as pd

class ResampleIntensityStage(Stage):
    REQUIRES = ["abundance", "species_info"]
    
    def __init__(self, distrib_type: str|Distribution = "normal", **kwargs):
        distrib_type, args = DISTRIBUTIONS[distrib_type]
        
        self.distrib_type = distrib_type
        self.dist_args = []
        for argname, _ in args:
            # Manually add all the distribution parameters to this stage's
            # param list
            param = StageParameter(Distribution,
                                   default=NormalDistribution(0, 1),
                                   msg_parser=dist_parser)
            self._params[argname] = StageParameter.Instance(param, argname)
            self.dist_args.append(argname)
            
        super().__init__(**kwargs)
        
    def default_name(self):
        return "resample_intensity"
    
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        assert input.abundance is not None
        new_abundance = np.zeros_like(input.abundance)

        for samp_id in range(input.abundance.shape[0]):
            intens = input.abundance[samp_id, :]
            presence = intens > 0
            # Get all non-zero abundances and their sorted order
            pos_vals = intens[presence]
            pos_sort = np.argsort(pos_vals)

            # Create the distribution for this sample.
            params = {argname:state[argname](N=1, rng=rng)
                      for argname in self.dist_args}
            dist = self.distrib_type(**params)
            # Draw the same number of random intensities and sort
            new_vals = dist(presence.sum(), rng=rng)
            new_vals.sort()

            # Now, reorder the new_vals so that they are in the same order as the
            # original
            new_vals = new_vals[pos_sort]
            new_abundance[samp_id, presence] = new_vals
        return PipelineData(
            input.copy(abundance=new_abundance),
            messages=[], rng=rng)


class IntensityFilterStage(Stage):
    REQUIRES = ["abundance", "species_info"]

    min_intensity = StageParameter(float, 1e4)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def default_name(self):
        return "filter_intensity"
    
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        assert input.abundance is not None
        low = input.abundance < state.min_intensity
                
        new_abundance = input.abundance.copy()
        new_abundance[low] = 0
        return PipelineData(
            input.copy(abundance=new_abundance),
            messages=[], rng=rng)
