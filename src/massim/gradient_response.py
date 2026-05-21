import pandas as pd
import numpy as np
from typing import Sequence
from abc import ABC, abstractmethod

from .utils import dotdict
from .distributions import (
    RNG, gen_param_dist, Distribution, UniformDistribution, NormalDistribution, Clump
)


class GradientResponse(ABC):
    class ResponseSet(ABC):

        def __init__(self, config, index, params):
            self.config = config
            self.index = index
            self.params = params
            
        @abstractmethod
        def apply(self, X: np.ndarray) -> np.ndarray:
            pass


    def __init__(self, param_dists: dict[str, Distribution]):
        self.param_dists = param_dists

    def _generate_vals(self, index: list, rng: np.random.Generator, validator=None):
        """
        If validator is given, it should be a func taking arrays as
        input (one arg per param), and return a boolean mask with True where
        the values need to be regenerated
        """
        N = len(index)
        invalid = np.ones(N, dtype=np.bool)
        result = {k: np.zeros(N) for k in self.param_dists.keys()}
        
        while invalid.any() :
            N = invalid.sum()
            for k, v_dist in self.param_dists.items():
                result[k][invalid] = v_dist(N, rng)
            if validator is None:
                invalid *= 0
            else:
                invalid = validator(**result)

        return result
        
    @abstractmethod
    def generate_response(self, index: list,
                          rng: np.random.Generator | None = None
                          ) -> ResponseSet:
        pass


class BetaResponse(GradientResponse):
    class ResponseSet(GradientResponse.ResponseSet):
        """Class for response curves"""
        def __init__(self, config, species, **params):
            self.config = config
            self.species = species
            self.m = params["mode_"]
            self.r = params["range"]
            self.alpha = params["alpha"]
            self.gamma = params["gamma"]

        def duplicate(self, beta_mul=1.0):
            return BetaResponse.ResponseSet(
                config=self.config,
                species=self.species,
                mode_=self.m,
                range=self.r,
                alpha=beta_mul*self.alpha,
                gamma=beta_mul*self.gamma)

        def beta(self, x):
            """Beta function: x^alpha * (1-x)^gamma"""
            return x**self.alpha * (1 - x)**self.gamma

        def df(self):
            return pd.DataFrame(dict(
                mode=self.m,
                range=self.r,
                alpha=self.alpha,
                gamma=self.gamma),
                                
                index=self.species)

        def area(self):
            import scipy.special as SS
            b = self.alpha / (self.alpha + self.gamma)
            d = b**self.alpha * (1-b)**self.gamma
            # Integration for area under the curve from Mathematica
            return (self.r / d
                    * SS.gamma(1+self.alpha) * SS.gamma(1+self.gamma)
                    / SS.gamma(2 + self.alpha + self.gamma))

        def equalize(self, A0, major_frac):
            """
            Find the top major species along this gradient and relocate them
            to be evenly spaced
            """
            m = int(len(self.m) * major_frac)
            if m == 0:
                return
            T = self.area() * A0
            # Indices of top m species
            majors = T.argpartition(-m)[-m:]
            # define fraction of gradient for each
            tot = 100 * T[majors] / T[majors].sum()
            # New locations at midpoint of each partition
            # Note that argpartition doesn't sort the top 'm'
            new_m = tot.cumsum() - tot / 2
            self.m[majors] = new_m
            
        
        def apply(self, X, beta_mul=1.0):
            alpha = self.alpha * beta_mul
            gamma = self.gamma * beta_mul

            def beta(x):
                """Beta function: x^alpha * (1-x)^gamma"""
                return x**alpha * (1-x)**gamma

            b = alpha / (alpha + gamma)
            d = beta(b)
            range_lo = self.m - self.r * b
            range_hi = self.m + self.r * (1-b)

            if isinstance(X, list):
                X = np.array(X)[:, None]
            elif isinstance(X, np.ndarray):
                X = X[:, None]
            temp = (X - self.m)/self.r + b
            temp[X < range_lo] = 0
            temp[X > range_hi] = 0        
            prod = beta(temp)
            return prod /d

    def __init__(self,
                 range=None,
                 mode=None,
                 alpha=None,
                 gamma=None,
                 no_skew=False):
        # Set alpha == gamma
        self.no_skew = no_skew
        if range is None:
            self._range = gen_param_dist("range")
        else:
            self._range = range
        if mode is None:
            self._mode = gen_param_dist("mode")
        else:
            self._mode = mode
        if alpha is None:
            self._alpha = gen_param_dist("alpha")
        else:
            self._alpha = alpha
        if gamma is None:
            self._gamma = gen_param_dist("gamma")
        else:
            self._gamma = gamma
        super().__init__(dict(range=self._range,
                              mode_=self._mode,
                              alpha=self._alpha,
                              gamma=self._gamma,))
                              

    def __repr__(self):
        return (f"GradientResponse(m={self.mode}, r={self.range}, "
                f"alpha={self.alpha}, gamma={self.gamma} "
                f"no_skew={self.no_skew})")

    @property
    def state(self):
        params = dotdict(
            no_skew=self.no_skew,
            mode=self.mode,
            range=self.range,
            alpha=self.alpha,
            gamma=self.gamma
        )
        return State("GradientResponse.Config", "", params)

    @property
    def range(self):
        return self._range

    @property
    def mode(self):
        return self._mode

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    def adjust(self, mode=0, range=0, beta=0):
        m = self.mode.widen(mode)
        r = self.range.increase(range).widen(range)
        a = self.alpha.decrease(beta)
        if self.gamma:
            g = self.gamma.decrease(beta)
        else:
            g = None
        return BetaResponse(mode=m, range=r, alpha=a, gamma=g,
                            no_skew=self.no_skew)

    def generate_response(self, index: list, rng=None):
        rng = RNG(rng)
        N = len(index)

        # We want to ensure that all species exist *somewhere* within the
        # gradient, meaning that the mode +/- range has to intersect
        # [0-100]
        # If not, we flag for recomputation.
        def validator(mode_, range, alpha,  gamma):
            b = alpha / (alpha + gamma)
            return (mode_ + range * (1-b) <=0) | (mode_ - b * range >= 100)
            
        params = self._generate_vals(index, rng, validator)
        
        
        if self.no_skew:
            params['gamma'] = params['alpha']

        return BetaResponse.ResponseSet(
            self,
            index,
            **params
        )

    @staticmethod
    def SimpleResponse(mean_mode,
                       mean_range,
                       sd_range=30,
                       shape=0.5,
                       skewness=0):
        alpha_mean = (1 + skewness) * shape
        gamma_mean = (1 - skewness) * shape
        return BetaResponse(
            mode=UniformDistribution(
                min=(mean_mode - 100),
                max=(mean_mode + 100)),
            range=NormalDistribution(
                mean=mean_range,
                std=sd_range),
            alpha=UniformDistribution(
                min=alpha_mean/4,
                max=alpha_mean*2),
            gamma=UniformDistribution(
                min=gamma_mean/4,
                max=gamma_mean*2),
            no_skew=(skewness == 0))

    @staticmethod
    def DefaultResponse(mode=50, beta_div=1, clump=1):
        return BetaResponse(
            mode=Clump(mode-100, mode+100, clump=clump),
            range=NormalDistribution(100/beta_div, 30/beta_div),
            alpha=UniformDistribution(2.5, 6.5),
            gamma=UniformDistribution(2.5, 6.5),
            no_skew=False)
    

# For simplicities sake, all gradients range from 0-100
class LinearResponseConfig(GradientResponse):
    class ResponseSet(GradientResponse.ResponseSet):
        def __init__(self, config, index, center, rise):
            self.config = config
            self.index = index
            self.center = center
            self.rise = rise

        def df(self):
            return pd.DataFrame(
                dict(
                    center=self.center,
                    rise=self.rise),
                index=self.index)

        def apply(self, X):
            if isinstance(X, list):
                X = np.array(X)[:, None]
            elif isinstance(X, np.ndarray):
                X = X[:, None]
            return self.center + (X - 50)/50 * self.rise

    def __init__(self, center, rise):
        """Define a linear response by defining the value at
        gradient center (50), and the rise at gradient max(100).
        """
        self.center = center
        self.rise = rise

    def generate_response(self, index, rng=None):
        rng = RNG(rng)
        N = len(index)
        return LinearResponseConfig.ResponseSet(
            self,
            index,
            self.center(N, rng),
            self.rise(N, rng))


class LandscapeResponse(ABC):
    class ResponseSet(ABC):
        def __init__(self, config, index, params, responses):
            self.config = config
            self.index = index
            self.responses = responses
            self.params = params

        def __getitem__(self, key):
            if key in self.params:
                return self.params[key]
            return self.responses[key]

        @abstractmethod
        def apply(self, coords: pd.DataFrame) -> np.ndarray:
            pass


    def __init__(self,
                 name: str,
                 index_name: str,
                 column_name: str,
                 allow_missing_gradients: bool=False,
                 **params):
        self.name = name
        self.responses: dict[str, GradientResponse] = {}
        self.params: dict[str, Distribution] = params
        self.allow_missing_gradients = allow_missing_gradients
        
    @abstractmethod
    def generate_response(self, index: list,
                          rng: np.random.Generator | None = None
                          ) -> ResponseSet:
        pass

    def set_gradient_response(self,
                              grad_id: str,
                              response: GradientResponse):
        self.responses[grad_id] = response

    def set_param(self, param_id: str, dist: Distribution):
        self.params[param_id] = dist

    def check_gradients(self, grads: Sequence[str]):
        if not self.allow_missing_gradients:
            for grad in grads:
                if grad not in self.responses:
                    raise ValueError(f"Mapper '{self.name}' contains no "
                                     f"response for gradient '{grad}'.")

        for grad in self.responses:
            if grad not in grads:
                raise ValueError(f"Mapper '{self.name}' contains a"
                                 "response for unknown gradient "
                                 f"'{grad}'.")

    def _generate_vals(self, index: list,
                       rng: np.random.Generator | None = None):
        param_vals = {}
        N = len(index)
        for param_name, param in self.params.items():
            if callable(param):
                param_vals[param_name] = param(N, rng)
            elif not hasattr(param, '__iter__'):
                param_vals[param_name] = param * np.ones(N)
            else:
                param_vals[param_name] = param

        response_set = {k: r.generate_response(index, rng)
                        for k, r in self.responses.items()}
        return param_vals, response_set


    @abstractmethod
    def _combine_gradients(self,
                           grad_results: dict[str, np.ndarray]
                           ) -> pd.DataFrame:
        pass

    def apply(self, coords: pd.DataFrame):
        grad_results = {}
        for grad_id in coords.columns:
            response = self.responses.get(grad_id)
            if response is None and not self.allow_missing_gradients:
                raise ValueError(f"Missing coordinates for gradient {grad_id}")
            grad_results[grad_id] = response.apply(coords[grad_id].values)
