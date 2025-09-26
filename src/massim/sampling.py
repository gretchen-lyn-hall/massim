from __future__ import annotations

import pandas as pd
import numpy as np
from collections import namedtuple
from collections.abc import Callable
from abc import ABC, abstractmethod
import re

from .utils import dotdict

from .distributions import Distribution, RNG, UniformDistribution
from .experiment import Stage, StageParameter, StageData, PipelineData

TILE = 1
REPEAT = 2
INFO_RE = re.compile(r"^\*(.+)\*$")


class SamplerConfig(ABC):
    """
    The overall goal of SamplerConfig and its subclasses is to provide a
    flexible way to specify sampling across the virtual landscape.
    A very sampler would sample a uniform grid across N gradients. However,
    it may be useful to sample some regions at a different resolution, or
    to allow for random variation.

    Thus, different sampling strategies can be specified as SampleGroups.
    Each SampleGroup specifies the layout of samples across all gradients,
    composed of some combindation of GridSampler and RandomSampler.
    (SampleGroupCollection is a top-level container for SampleGroups)
    
    The key interface is 'sample()' which returns a dataframe of sample
    coordinates.
    """
    class SampleSet:
        """Class for returning sample locations from subsamplers.
        When a subsampler is invoked, it returns a list of coordinates for
        a (possible) subset of the gradients. This class allows partial
        sample coordinates to be tiled and aligned.
        """
        def __init__(self):
            self.data = pd.DataFrame()

        @property
        def keys(self) -> list[str]:
            return [c for c in self.data.columns]

        def __getitem__(self, key: str) -> np.ndarray:
            if key in self.keys:
                return self.data[key].values
            raise KeyError(f"'{key}' not in SampleSet")

        def merge(self, rhs: SamplerConfig.SampleSet) -> None:
            if len(self) == 0:
                self.data = rhs.data.copy()
                return
            
            if len(self) != len(rhs):
                raise ValueError("Merging SampleSets of unequal length")
            common_keys = set(self.keys).intersection(rhs.keys)
            if common_keys:
                common_keys = ", ".join(common_keys)
                raise ValueError("Can't merge SampleSets: duplicate keys: "
                                 f"{common_keys}")
            self.data = pd.concat([self.data, rhs.data], axis=1)

        def append(self, rhs: SamplerConfig.SampleSet) -> None:
            if len(self) == 0:
                self.data = rhs.data.copy()
                return
            diff = set(self.keys).symmetric_difference(rhs.keys)
            if diff:
                raise ValueError("Appended SampleSets must share all keys")
            self.data = pd.concat([self.data, rhs.data], axis=0)

        def extend(self, times: int, mode=TILE) -> None:
            if mode == TILE:
                self.data = pd.concat([self.data] * times, ignore_index=True)
            else:
                idx = self.data.index.repeat(times)
                self.data = self.data.loc[idx].reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.data)

        def insert_data(self, key: str, data: np.ndarray):
            if key in self.keys:
                raise ValueError(f"Adding existing key '{key}'")

            if len(self.data) > 0 and len(data) != len(self.data):
                raise ValueError("Adding data of incompatible length")
            self.data[key] = data

        def df(self, index=None) -> pd.DataFrame:
            """Return the sampled coordinates in a pandas DataFrame

            The result will have the gradient ids as columns
            """
            return pd.DataFrame(self.data.values,
                                columns=self.data.columns,
                                index=index)

    """Base class used to generate samples along gradients."""
    def __init__(self, name: str, N: int, adjustable: bool = False):
        self.name = name
        self._N = N
        self.adjustable = adjustable

    def check_gradient(self, grad_id, expect: bool|None = None) -> bool:
        """Check that the sampler does/does not sample the named gradient.

        If 'expect' is None, returns whether the sampler samples the gradient.
        Otherwise, raise an error if the sampler does/does not sample their
        gradient, depending on the value of 'expect'
        """
        
        result = grad_id in self.sampled_gradients
        if expect is not None and result != expect:
            if expect is True:
                raise ValueError(f"Unknown gradient '{grad_id}'")
            else:
                raise ValueError(f"Gradient '{grad_id}' already exists.")
        return result

    def sample_ids(self, N: int, labelfunc: Callable|None = None):
        """Generate a list of N string ids for the sample.
        If `labelfunc` is supplied, it should be a function taking a row
        of the sample_info dataframe and returning a string.
        """
        
        def default_labeler(row):
            tag = "_".join(str(x) for x in row)
            return "s_"+tag

        info = self._info(N).df()
        if labelfunc is None:
            labelfunc = default_labeler
        sids = info.apply(labelfunc, axis=1)
        return sids.values

    @property
    @abstractmethod
    def sampled_gradients(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def N(self) -> int:
        return self._N

    @abstractmethod
    def adjust_n(self, N: int) -> int:
        """Adjust the number of samples produced by the sampler.
        Input is the new target number of samples. A sampler can either
        accept this, or output another value >= N that will work for this
        sampler (for instance, the next multiple of a grid size)
        """
        
        pass
    
    @abstractmethod
    def _sample(self, N: int|None,
                rng: np.random.Generator|None
                ) -> SamplerConfig.SampleSet:
        pass

    @abstractmethod
    def _info(self, N=None, rng=None)-> SamplerConfig.SampleSet:
        pass

    def sample(self, rng: np.random.Generator|None = None) -> pd.DataFrame:
        """Return coordinates sampled from the landscape
        The resulting dataframe is indexed by sample_id and contains one
        column per coordinate.
        """
        result_set = self._sample(self.N, rng=rng)
        return result_set.df(index=self.sample_ids(self.N))

    def info(self):
        """Return a pandas DataFrame containing information about the sample
        locations.
        The result is indexed by sample id, and the type of information depends
        on the underlying subsamplers.
        """
        result_set = self._info(self.N)
        return result_set.df(index=self.sample_ids(self.N))


class SampleGroupCollection(SamplerConfig):
    def __init__(self, name):
        self.subgroups = []
        # Sample size MUST be determined by subgroups:
        super().__init__(name=name, N=None)

    def adjust_n(self, N):
        result = 0
        for subg in self.subgroups:
            result += subg.adjust_n(N)
        return result

    @property
    def sampled_gradients(self):
        result = set()
        for subg in self.subgroups:
            result = result.union(subg.sampled_gradients)
        return result

    @property
    def state(self):
        sub_states = [grp.state for grp in self.subgroups]
        return State("SampleGroupCollection", self.name, sub_states)

    @property
    def N(self):
        result = super().N
        return sum(subg.N for subg in self.subgroups)

    def add_subgroup(self, sampler):
        if not isinstance(sampler, SamplerConfig):
            raise ValueError("Argument is not a SamplerConfig object")

        diff = self.sampled_gradients.symmetric_difference(sampler.sampled_gradients)

        if len(self.subgroups) > 0 and len(diff) > 0:
            diff = ", ".join(gr for gr in diff)
            raise ValueError("Subgroups contain different gradients: "
                             f"{diff}")
        self.subgroups.append(sampler)
        return self

    def _sample(self, N=None, rng=None):
        if N is None:
            N = self.N

        result = SamplerConfig.SampleSet()
        for subg in self.subgroups:
            subsamp = subg._sample(rng=rng, N=None)
            result.append(subsamp)
        return result

    def _info(self, N=None):
        if N is None:
            N = self.N

        result = SamplerConfig.SampleSet()
        for subg in self.subgroups:
            subinfo = subg._info(N=None)
            subinfo.insert_data(self.name, [subg.name] * len(subinfo))
            result.append(subinfo)

        return result


class SampleGroup(SamplerConfig):
    def __init__(self, name, N=None, shortname=None):
        super().__init__(name, N=N)
        if shortname is not None:
            self.shortname = shortname
        else:
            self.shortname = self.name[:2]
        self.samplers = []

    def adjust_n(self, N):
        for subg in self.samplers:
            N = subg.adjust_n(N)
        return N

    @property

    def sampled_gradients(self):
        result = set()
        for subg in self.samplers:
            result = result.union(subg.sampled_gradients)
        return result

    @property
    def N(self):
        result = super().N
        sub_ns = [subg.N for subg in self.samplers if subg.N is not None]
        if sub_ns:
            result = max(sub_ns)
            
        result = self.adjust_n(result)

        if result != self.adjust_n(result):
            raise ValueError("Can not automatically adjust sample size for "
                             f"SampleGroup '{self.name}'")
        return result
        
    @property
    def state(self):
        sub_states = [grp.state for grp in self.samplers]
        return State("SampleGroup", self.name, sub_states)

    def add_sampler(self, sampler):
        if not isinstance(sampler, SamplerConfig):
            raise ValueError("Argument is not a SamplerConfig object")
        inter = self.sampled_gradients.intersection(sampler.sampled_gradients)
        if len(inter) > 0:
            raise ValueError("Sampler already samples gradients "
                             f"{', '.join(inter)}")
        
        self.samplers.append(sampler)
        return self

    def _sample(self, N=None, rng=None):
        if N is None:
            N = self.N

        result = SamplerConfig.SampleSet()
        for subg in self.samplers:
            subsamp = subg._sample(rng=rng, N=N)
            result.merge(subsamp)
        return result

    def _info(self, N=None):
        if N is None:
            N = self.N

        result = SamplerConfig.SampleSet()
        for subg in self.samplers:
            subinfo = subg._info(N=N)
            result.merge(subinfo)

        return result

    
class GridSampler(SamplerConfig):
    GridGradient = namedtuple("GridGradient", ['grad_id', 'N', 'lo', 'hi'])
    
    def __init__(self, name, N=None):
        super().__init__(name, N=N)
        self.dimensions = []
        self.grid_size = 0

    def add_gradient(self, grad_id, N, lo=0, hi=100):
        self.check_gradient(grad_id, expect=False)
        new_dim = GridSampler.GridGradient(grad_id, N, lo, hi)
        self.dimensions.append(new_dim)
        if self.grid_size == 0:
            self.grid_size = N
        else:
            self.grid_size *= N
        return self

    @property
    def state(self):
        return State("GridSampler", self.name, self.dimensions)

    @property
    def sampled_gradients(self):
        return set(dim.grad_id for dim in self.dimensions)

    @property
    def N(self):
        if super().N is None:
            return self.grid_size
        return super().N

    def _mesh(self, arrays, tile=1):
        result = SamplerConfig.SampleSet()
        meshed = np.meshgrid(*arrays)
        for dim, mesh in zip(self.dimensions, meshed):
            result.insert_data(dim.grad_id,
                               np.tile(mesh.flatten(), tile))
        return result
    
    def adjust_n(self, N):
        # Theoretically, should return lcm(N, gridsize), but that might
        # lead to a husg number of samples.
        # Instead, we'll adjust to nearedt multiple
        if N % self.grid_size == 0:
            return N
        return self.grid_size * (N // self.grid_size + 1)
        
    def _sample(self, N, rng=None):
        if N % self.grid_size != 0:
            raise ValueError(f"N ({N}) is not a multiple of grid size "
                             f"({self.grid_size}).")

        tile = N // self.grid_size
        result = self._mesh([np.linspace(d.lo, d.hi, d.N)
                             for d in self.dimensions], tile=tile)
        return result

    def _info(self, N, rng=None):
        if N % self.grid_size != 0:
            raise ValueError(f"N ({N}) is not a multiple of grid size "
                             f"({self.grid_size}).")

        tile = N // self.grid_size
        result = self._mesh([range(d.N) for d in self.dimensions], tile=tile)
        return result


class PerturbationSampler(SamplerConfig):
    def __init__(self, name, n_steps, subsampler, n_interp=0):
        super().__init__(name, N=subsampler.N * (n_steps * (1+n_interp)))
        self.subsampler = subsampler
        self.n_steps = n_steps
        self.n_interp = n_interp
        self.gradients = {}

    def add_gradient(self, grad_id, dist):
        if grad_id not in self.subsampler.sampled_gradients:
            raise ValueError(f"Gradient {grad_id} not in subsampler.")
        self.gradients[grad_id] = dist

    @property
    def substeps(self):
        # If interpolation is enabled, then between any two steps, we
        # have n_interp extra
        return self.n_steps + (self.n_steps - 1) * self.n_interp

    @property
    def N(self):
        return self.subsampler.N * self.substeps
        
    @property
    def sampled_gradients(self) ->list[str]:
        return self.subsampler.sampled_gradients()

    def adjust_n(self, N):
        return self.N

    @staticmethod
    def spline_4p( t, p_1, p0, p1, p2 ):
        """Basic catmull-rom spline for interpolating between p0 and p1.
        """
        return (
              t*((2-t)*t - 1)   * p_1
            + (t*t*(3*t - 5) + 2) * p0
            + t*((4 - 3*t)*t + 1) * p1
            + (t-1)*t*t         * p2 ) / 2    

    def _interp(self, vecs):
        # Perform a catmul-rom spline
        # We'll handle interpolation near endpoints by doubling up the endpoints
        # which essentially gives us the difference P'(0) = (P1-P0)/2
        idxs = [0] + list(range(len(vecs))) + [len(vecs)-1]
        vecpad = vecs[idxs]

        # The t values (in (0-1)) for each inner interpolation
        # Ignore start point too, and convert to column vector
        inner = np.linspace(0, 1.0,
                            self.n_interp + 1,
                            endpoint=False)[1:][:, None]

        result = np.zeros((self.substeps, vecs.shape[1]))
        per_step = self.n_interp + 1
        for idx, vec in enumerate(vecs[:-1]):
            interp = self.spline_4p(inner,
                                    vecpad[idx],
                                    vecpad[idx+1],
                                    vecpad[idx+2],
                                    vecpad[idx+3])
            cur_loc = idx * per_step
            result[cur_loc] = vec
            result[cur_loc+1:cur_loc+per_step] = interp

        result[-1] = vecs[-1]
        return result
        

        
        

    def _sample(self, N: int|None,
                rng: np.random.Generator|None
                ) -> SamplerConfig.SampleSet:
        if N is None:
            N = self.N
        if N != self.N:
            raise ValueError(f"N ({N}) is not equal to {self.N} ")
        rng = RNG(rng)

        subsamp = self.subsampler.sample(rng=rng)
        subvals = subsamp.values
        # For each step, pick a random vector to add to subsamp
        vecs = []
        for grad_id in subsamp.columns:
            dist = self.gradients.get(grad_id)
            if dist is not None:
                vecs.append(dist(self.n_steps, rng=rng))
            else:
                vecs.append(np.zeros(self.n_steps))
        vecs = np.stack(vecs, axis=1)
        result_mat = np.concat([subvals + vecs[ii]
                                for ii in range(vecs.shape[0])])

        if self.n_interp > 0:
            result_mat = self._interp(result_mat)

            

        result = SamplerConfig.SampleSet()
        for idx, grad_id in enumerate(subsamp.columns):
            result.insert_data(grad_id, result_mat[:, idx])
        return result
    
    def _info(self, N=None, rng=None)-> SamplerConfig.SampleSet:
        subinfo = self.subsampler.info()

        subvals = np.tile(subinfo.values,
                          [self.substeps, 1])
        result = SamplerConfig.SampleSet()
        for idx, col in enumerate(subinfo.columns):
            result.insert_data(col, subvals[:, idx])
            
        # between each two steps we have n_interp substeps, capped off by
        # the last step.
        steps = np.append(
            np.repeat(np.arange(self.n_steps - 1), self.n_interp + 1),
            [self.n_steps - 1])
        result.insert_data(f"{self.name}_step", steps)

        if self.n_interp > 0:
            interp = np.tile(np.arange(self.n_interp + 1), self.n_steps - 1)
            # Add 0 for the last step
            interp = np.append(interp, 0)
            result.insert_data(f"{self.name}_interp", interp)
            
        return result
        
        
        
class RandomSampler(SamplerConfig):
    def __init__(self, name,  N=None):
        super().__init__(name, N=N)
        self.gradients = {}

    @property
    def N(self):
        return self._N

    @property
    def state(self):
        return State("RandomSampler", self.name,
                     [(grad_id, dist.state)
                      for grad_id, dist in self.gradients.items()])

    @property
    def sampled_gradients(self):
        return set(self.gradients)

    def adjust_n(self, N):
        return N

    def add_gradient(self, grad_id, dist):
        if not isinstance(dist, Distribution):
            raise ValueError("Argument must be a valid distribution")
        self.check_gradient(grad_id, expect=False)
        self.gradients[grad_id] = dist
        return self

    def _sample(self, N, rng=None):
        rng = RNG(rng)
        if N is None:
            N = self.N
        result = SamplerConfig.SampleSet()
        for grad_id, dist in self.gradients.items():
            result.insert_data(grad_id, dist(N, rng=rng))
        return result

    def _info(self, N, rng=None):
        rng = RNG(rng)
        if N is None:
            N = self.N
        result = SamplerConfig.SampleSet()
        for grad_id, dist in self.gradients.items():
            result.insert_data(grad_id, range(N))
        return result



class TransectSampler(SamplerConfig):
    METHODS = ["even",   # Sample at evenly spaced locations along transect
               "random", # Sample randomly across transect
               ]
    def __init__(self, name,  N, sampling_method="even"):
        super().__init__(name, N=N)
        self.gradients = {}
        if sampling_method not in ["even", "random"]:
            raise ValueError(f"Sampling method must be one of: {self.METHODS}.")
        self.sampling_method = sampling_method

    @property
    def N(self):
        return self._N

    @property
    def sampled_gradients(self):
        return set(self.gradients)

    def adjust_n(self, N):
        return N

    def add_gradient(self, grad_id, start_val, end_val):
        self.check_gradient(grad_id, expect=False)
        self.gradients[grad_id] = (start_val, end_val)
        return self

    def _sample(self, N, rng=None):
        rng = RNG(rng)
        if N is None:
            N = self.N
        if self.sampling_method == "even":
            fracs = np.linspace(0, 1, N)
        else:
            fracs = UniformDistribution(0, 1)(N, rng=rng)
            fracs.sort()
            
        result = SamplerConfig.SampleSet()
        for grad_id, (from_val, to_val) in self.gradients.items():
            result.insert_data(grad_id, from_val + (to_val - from_val) * fracs)
        return result

    def _info(self, N, rng=None):
        rng = RNG(rng)
        if N is None:
            N = self.N
        result = SamplerConfig.SampleSet()
        for grad_id, _ in self.gradients.items():
            result.insert_data(grad_id, range(N))
        return result


class GenSampleCoordsStage(Stage):
    PROVIDES = ["sample_coords", "sample_info"]

    def default_name(self) -> str:
        return "GenCoords"
    
    def __init__(self, samplers: SamplerConfig|list[SamplerConfig], **kwargs):
        super().__init__(**kwargs)
        if isinstance(samplers, SamplerConfig):
            self.sampler = samplers
        else:
            self.sampler = SampleGroupCollection("")
            for sampler in samplers:
                self.sampler.add_subgroup(sampler)


    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        sample_coords = self.sampler.sample(rng=rng)
        sample_info = self.sampler.info()

        data = input.copy(
            sample_coords=sample_coords,
            sample_info=sample_info);
        return PipelineData(data, [], rng)
        
        
