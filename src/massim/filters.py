from .experiment import Stage, PipelineData, StageData, StageParameter, dist_parser
from .distributions import (DISTRIBUTIONS,
                            Distribution,
                            UniformDistribution,
                            LogRandomDistribution,
                            NormalDistribution)


import numpy as np
import pandas as pd

class ResampleIntensityStage(Stage):
    """
    Replaces all the intensity values with values sampled from a distribution.
    Within each sample, the relative order of the values are maintained;
    that is the values drawn from the distribution are sorted, and each
    intensity is replaced by one of the same rank.

    In order to handle variations between samples. ResampleIntensityStage can
    take a family of distributions as an input. For each sample, a distribution
    is created with parameters drawn from their own user specified distributions.
    This is best explained by example.
    If we want intensities to be normally distributed, with mean uniformly
    distributed between 4 and 5 and unit standard deviation, we can use:
    >>> ResampleIntensityStage("normal", mean=UniformDistribution(4,5), std=1)
    
    """
    REQUIRES = ["abundance", "species_info"]
    
    def __init__(self, distrib_type: str|Distribution = "normal", **kwargs):

        if isinstance(distrib_type, Distribution):
            self.distrib_type = None
            self.dist = distrib_type
        else:
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
            self.dist = None

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
            # The "double-argsort" is used to map
            # position in sorted list -> original position.
            pos_unsort = np.argsort(pos_sort)

            if self.distrib_type is None:
                dist = self.dist
            else:
                # Create the distribution for this sample.
                params = {argname: state[argname](N=1, rng=rng)
                          for argname in self.dist_args}
                dist = self.distrib_type(**params)
                
            # Draw the same number of random intensities and sort
            new_vals = dist(presence.sum(), rng=rng)
            new_vals.sort()

            # Now, reorder the new_vals so that they are in the same order as the
            # original
            new_vals = new_vals[pos_unsort]
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




class AdaptiveIntensityThreshold(Stage):
    """
    An adaptive filter that sets an intensity threshold so that, on average,
    each sample contains a specified fraction of non-zero values.
    Can be used to set average number of metabolites per sample by
    setting `target_presence` to (# desired) / (# metabolites)
    
    """
    REQUIRES = ["abundance", "species_info"]

    target_presence = StageParameter(float, 0.1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def default_name(self):
        return "filter_intensity"
    
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        assert input.abundance is not None
        thresh = np.quantile(input.abundance.flatten(), 1 -state.target_presence)
        
        new_abundance = input.abundance.copy()
        new_abundance[new_abundance < thresh] = 0
        return PipelineData(
            input.copy(abundance=new_abundance),
            messages=[], rng=rng)


