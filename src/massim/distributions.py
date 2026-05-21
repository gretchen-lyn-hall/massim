from __future__ import annotations
from collections import namedtuple
from typing import Iterable, Callable
from dataclasses import dataclass
import inspect
import re

import numpy as np
import pandas as pd

FP_RE = r"(?:[-+]?\d*\.?\d+)(?:[eE](?:[-+]?\d+))?"
FUNC_RE = re.compile(f"^(\\w+)\\(({FP_RE}(?:,{FP_RE})*)\\)$")
LIST_RE = re.compile(r"^\[(-?[\d.]+(?:,-?[\d.]+)*)\]$")

from .utils import dotdict


def RNG(rng_or_seed: int|np.random.Generator|None=None) -> np.random.Generator:
    """Return a random number generator based off of the argument.


    Parameters
    ----------
    rng_or_seed : None, int, or np.random.Generator
         None: Return a new random generator with a random seed.
         int: Return a new random generator with given seed
         np.random.Generator: Spawn a new generator from the argument.

    Returns
    -------
    out: np.random.Generator
    """
    if rng_or_seed is None:
        return np.random.default_rng()
    if isinstance(rng_or_seed, int):
        return np.random.default_rng(rng_or_seed)
    return rng_or_seed.spawn(1)[0]



class TestRNG:
    @dataclass
    class DistRecord:
        data: Iterable|Callable
        index: int = 0
    
    def __init__(self, repeat=True, default=None, **kwargs):
        """Return canned responses instead of random values.
        The responses for each distribution (uniform, normal, etc)
        should be specified in kwargs; e.g.:
          TestRNG(uniform=[1,2,3])
        will return 1, 2, and 3
        If 'repeat' is True, the values in the list will be cycled through,
        otherwise an error will be raised.
        The value must be either an iterable or a function. If a function,
        it will be called with arguments `N` and `index`.
        
        If 'default' is specified, it should be an iterable or function that
        will be used for all unspecified distributions
        """
        self.repeat = repeat
        # Dict of mock distributions, and the current position
        self.responses = {}
        for dist_name, data in kwargs.items():
            if not isinstance(data, (Iterable, Callable)):
                raise TypeError(f"Mock distribution '{dist_name}' must be "
                                "iterable or a function.")
            self.responses[dist_name] = TestRNG.DistRecord(data)
        if default is not None:
            self.responses['default'] = TestRNG.DistRecord(default)
        self.log = []

    def history(self):
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.2}"
            return str(x)
               
        lines = []
        for item in self.log:
            kwargs = [f"{k}={v}" for k, v in item["kwargs"].items()]
            arglist = ", ".join(fmt(x) for x in list(item["args"]) + kwargs)
            line = f"[#={item['size']:4}]"
            line = line.ljust(15) + f"{item['dist_name']}({arglist})"
            lines.append(line)
        return '\n'.join(lines)

    def spawn(self, count):
        return [self for _ in range(count)]
            
    def _make_dist_call(self, dist_rec, dist_name):
        def call_dist(*args, size=1, **kwargs):
            self.log.append(dict(
                dist_name=dist_name,
                size=size,
                args=args,
                kwargs=kwargs))
            if isinstance(size, tuple):
                N = np.prod(size)
            else:
                N = size
                
            if isinstance(dist_rec.data, Callable):
                result = dist_rec.data(N, dist_rec.index)
                dist_rec.index += N
            else:
                mode = 'wrap' if self.repeat else 'raise'
                to_idx = dist_rec.index + N
                try:
                    result = np.take(dist_rec.data,          # type: ignore
                                     np.arange(dist_rec.index, to_idx),
                                     mode=mode)
                except IndexError as e:
                    raise IndexError("Not enough data for mock distribution")
                dist_rec.index = to_idx % len(dist_rec.data)
                
            return np.reshape(result, size)

        return call_dist

    def __getattr__(self, dist_name):
            dist_rec = self.responses.get(dist_name)
            if dist_rec is None:
                dist_rec = self.responses.get('default')
            if dist_rec is None:
                raise ValueError("No mock data specified for distribution "
                                 f"{dist_name}.")
            return self._make_dist_call(dist_rec, dist_name)
            
                


# Parameters, along with their default ranges
Parameter = namedtuple("Parameter",
                       [
                           "symbol",
                           "description",
                           "def_min",
                           "def_max",
                           "def_mean",
                           "def_var",
                           "def_dist",
                           ])

DISTRIBUTIONS = {}
DistInfo = namedtuple("DistInfo", ["cls", "args"])
ArgInfo = namedtuple("ArgInfo", ["name", "type"])

def distribution(cls):
    init_method = cls.__init__
    if not callable(init_method):
        return 0
    init_signature = inspect.signature(init_method)
    args = []
    for param in init_signature.parameters.values():
        if param.name in ["self", "defaults"]:
            continue
        typ = param.annotation
        if typ == inspect._empty:
            typ = None
        args.append(ArgInfo(param.name, typ))

    DISTRIBUTIONS[cls.__name__] = DistInfo(cls, args)
    DISTRIBUTIONS[cls] = DistInfo(cls, args)
    
    return cls

class Distribution:     
    def __init__(self, mapping, defaults=None, shortname=None, **kwargs):
        if shortname is None:
            cname = self.__class__.__name__.removesuffix("Distribution")
            self.shortname = cname.lower()
        else:
            self.shortname=shortname
        if isinstance(defaults, Parameter):
            defaults = defaults._asdict()
        if defaults is None:
            defaults = {}
        self._params = {}
        for k, def_name in mapping.items():
            if k in kwargs and kwargs[k] is not None:
                value = kwargs[k]
            else:                    
                if def_name not in defaults:
                    raise ValueError("Distribution is missing required "
                                     f"parameter '{k}'")
                value = defaults[def_name]
            self._params[k] = value
            setattr(self, k, value)
    
    def __repr__(self):
        plist = ", ".join(f"{k}={getattr(self, k)}" for k  in self._params)
        return f"{self.__class__.__name__}({plist})"

    def as_str(self):
        plist = ", ".join(str(getattr(self, k)) for k  in self._params)
        return f"{self.shortname}({plist})"

    @property
    def param_list(self):
        return list(self._params.values())



    def widen(self, percent):
        raise NotImplementedError()

    def narrow(self, percent):
        return self.widen(-percent)

    def increase(self, percent):
        raise NotImplementedError()

    def decrease(self, percent):
        return self.increase(-percent)

    def update(self, param, val):
        if param not in self._params:
            raise ValueError(f"Distribution '{self.shortname}' has no parameter '{param}'.")
        self._params[param] = val
        setattr(self, param, val)

    def to_cdist(self):

        from massim._core import gen_dist
        params = self.param_list
        return gen_dist(self.shortname, params)

    
    @property
    def state(self):
        params = dotdict(self._params)
        return State("distribution", self.__class__.__name__, params)
            

    def apply_defaults(self, val, default_name):
        if val is None:
            val = self.defaults[default_name]

    def __call__(self, N, rng=None):
        rng = RNG(rng)
        return self._sample(rng, N)

    def __neg__(self):
        return NegateDistribution(self)

    def __add__(self, rhs):
        if isinstance(rhs, (int, float)):
            rhs = ConstantDistribution(rhs)
        return SumDistribution([self, rhs])

    def __radd__(self, rhs):
        if isinstance(rhs, (int, float)):
            rhs = ConstantDistribution(rhs)
        return SumDistribution([self, rhs])

    def __sub__(self, rhs):
        if isinstance(rhs, (int, float)):
            rhs = ConstantDistribution(rhs)
        return SumDistribution([self, rhs], [1, -1])
    
    def __rsub__(self, rhs):
        if isinstance(rhs, (int, float)):
            rhs = ConstantDistribution(rhs)
        return SumDistribution([rhs, self], [1, -1])

    def __mul__(self, rhs):
        if isinstance(rhs, (int, float)):
            return ScaleDistribution(self, rhs)
        return ProductDistribution([self, rhs])

    def __rmul__(self, rhs):
        if isinstance(rhs, (int, float)):
            return ScaleDistribution(self, rhs)
        return ProductDistribution([self, rhs])



def parse_func(text, line_no=None):
    text = text.replace(" ","")
    m = FUNC_RE.match(text)
    if m is None:
        raise ValueError(f"Unable to parse distribution {text}.")
    func_name = m.group(1)
    text_args = m.group(2).split(',')
    args = []
    for idx, arg in enumerate(text_args):
        try:
            args.append(float(arg))
        except ValueError:
            raise ValueError(f"Could not parse argument {idx+1} "
                             "({arg}) as a float.")
    # Account for the 'defaults' argument in constructor
    
def parse_distribution(text, line_no=None):
    text = text.replace(" ","")
    m = FUNC_RE.match(text)
    if m is None:
        raise ValueError(f"Unable to parse distribution {text}.")
    dist_name = m.group(1)
    dist = DISTRIBUTIONS.get(dist_name)
    if dist is None:
        raise ValueError(f"Unknown distribution: {dist_name}.")
    text_args = m.group(2).split(',')
    args = []
    if len(text_args) != len(dist.args):
        raise ValueError(f"'{dist_name} expects {len(dist.args)}  "
                         f"args; received {len(text_args)}")
    for idx, (in_arg, dist_arg) in enumerate(zip(text_args, dist.args)):
        typ = dist_arg.type
        if typ is None:
            typ = float
        try:
            args.append(typ(in_arg))
        except ValueError:
            raise ValueError(f"Could not parse argument {idx+1} "
                             f"({in_arg}) as type {typ}.")
    # Account for the 'defaults' argument in constructor
    try:
        return dist.cls(*args, defaults=None)
    except ValueError:
        raise ValueError(f"Too few arguments to distribution '{dist_name}'.")
    except TypeError:
        raise ValueError(f"Too many arguments to distribution '{dist_name}'.")

def gen_dist(dist_name, *args):
    if dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{dist_name}'.")
    cls, dist_args =  DISTRIBUTIONS[dist_name](*args)
    return cls(*args)

@distribution
class ConstantDistribution(Distribution):
    def __init__(self,
                 value: float | None = None,
                 defaults=None):
        super().__init__(dict(value="def_mean"),
                         defaults,
                         **dict(value=value))

    def _sample(self, rng, N):
        return np.ones(N) * self.value

    def widen(self, percent):
        return self


@distribution
class LinearRampDistribution(Distribution):
    def __init__(self,
                 start: float | None = None,
                 stop: float | None = None,
                 defaults=None):
        super().__init__(dict(start="def_min", stop="def_max"),
                         defaults,
                         **dict(start=start, stop=stop))

    def _sample(self, rng, N):
        return np.linspace(self.start, self.stop, num=N)

    def widen(self, percent):
        width = (self.stop - self.start) * (100 + percent) / 100.0
        ctr = (self.start + self.stop) / 2
        return LinearRampDistribution(ctr - width / 2,
                                      ctr + width / 2)

    def increase(self, percent):
        width = (self.stop - self.start)
        ctr = (self.start + self.stop) / 2  * (100 + percent) / 100.0 
        return LinearRampDistribution(ctr - width / 2,
                                      ctr + width / 2)
        


@distribution
class UniformDistribution(Distribution):
    def __init__(self, min: float =None, max: float =None, defaults=None):
        super().__init__(dict(min="def_min", max="def_max"),
                         defaults,
                         **dict(min=min, max=max))

    def _sample(self, rng, N):
        return rng.uniform(self.min, self.max, size=N)

    def widen(self, percent):
        width = (self.max - self.min) * (100 + percent) / 100.0
        ctr = (self.min + self.max) / 2
        return UniformDistribution(ctr - width / 2,
                                   ctr + width / 2)


    def increase(self, percent):
        width = (self.max - self.min)
        ctr = (self.min + self.max) / 2 * (100 + percent) / 100.0
        return UniformDistribution(ctr - width / 2,
                                   ctr + width / 2)


@distribution
class Clump(Distribution):
    def __init__(self, min: float =None, max: float =None, clump: int = 1, defaults=None):
        super().__init__(dict(min="def_min", max="def_max", clump=1),
                         defaults,
                         **dict(min=min, max=max, clump=clump))
        self.clump = clump

    def _sample(self, rng, N):
        ms =  rng.uniform(self.min, self.max, size=N * self.clump)
        if self.clump > 1:
            ms = np.sort(ms)
            probs = np.linspace(1/(N*self.clump), 1, N * self.clump) ** self.clump
            probs = probs / probs.sum()
            ms = rng.choice(ms, size=N, p=probs, replace=False)
        return ms

    def widen(self, percent):
        width = (self.max - self.min) * (100 + percent) / 100.0
        ctr = (self.min + self.max) / 2
        return Clump(ctr - width / 2,
                     ctr + width / 2,
                     self.clump)


    def increase(self, percent):
        width = (self.max - self.min)
        ctr = (self.min + self.max) / 2 * (100 + percent) / 100.0
        return Clump(ctr - width / 2,
                     ctr + width / 2,
                     self.clump)




@distribution
class NormalDistribution(Distribution):
    def __init__(self, mean=None, std=None, defaults=None):
        super().__init__(dict(mean="def_mean", std="def_var"),
                         defaults,
                         **dict(mean=mean, std=std))

    def _sample(self, rng, N):
        return rng.normal(self.mean, self.std, size=N)

    def widen(self, percent):
        return NormalDistribution(self.mean, self.std * (100 + percent)/100)

    def increase(self, percent):
        return NormalDistribution(self.mean * (100 + percent) / 100,
                                  self.std)


@distribution
class LogNormalDistribution(Distribution):
    def __init__(self, mean=None, std=None, defaults=None):
        super().__init__(dict(mean="def_mean", std="def_var"),
                         defaults,
                         **dict(mean=mean, std=std))

    def _sample(self, rng, N):
        return rng.lognormal(self.mean, self.std, size=N)

    def widen(self, percent):
        return LogNormalDistribution(self.mean,
                                     self.std * (100 + percent)/100)

    def increase(self, percent):
        return LogNormalDistribution(self.mean * (100 + percent) / 100,
                                  self.std)

@distribution
class LogRandomDistribution(Distribution):
    def __init__(self, min=None, max=None, defaults=None):
        super().__init__(dict(min="def_min", max="def_max"),
                         defaults,
                         **dict(min=min, max=max))

    def _sample(self, rng, N):
        return np.exp(
            rng.uniform(
                np.log(self.min), np.log(self.max), size=N))

    def widen(self, percent):
        width = (self.max - self.min) * (100 + percent) / 100.0
        ctr = (self.min + self.max) / 2
        return LogRandomDistribution(ctr - width / 2,
                                     ctr + width / 2)


    def increase(self, percent):
        width = (self.max - self.min)
        ctr = (self.min + self.max) / 2 * (100 + percent) / 100.0
        return LogRandomDistribution(ctr - width / 2,
                                     ctr + width / 2)

@distribution
class LogBetaDistribution(Distribution):
    def __init__(self, logmin=0, logmax=1, alpha=1.0, gamma=1.0, defaults=None):
        super().__init__(dict(min="def_min", max="def_max"),
                         defaults,
                         **dict(min=min, max=max, alpha=alpha, gamma=gamma))
        self.logmin = logmin
        self.logscale = logmax - logmin
        self.alpha = alpha
        self.gamma = gamma

    def _sample(self, rng, N):
        return 10 ** (self.logmin +
            rng.beta(self.alpha, self.gamma, size=N) * self.logscale)

    def widen(self, percent):
        width = (self.logmax - self.logmin) * (100 + percent) / 100.0
        ctr = (self.logmin + self.logmax) / 2
        self.logmin = ctr - width / 2
        self.logmax = ctr + width / 2

    def increase(self, percent):
        width = (self.logmax - self.logmin)
        ctr = (self.logmin + self.logmax) / 2 * (100 + percent) / 100.0
        self.logmin = ctr - width / 2
        self.logmax = ctr + width / 2

@distribution
class PermutedValues(Distribution):
    def __init__(self, values):
        super().__init__(dict())
        self.values = values
        
    def _sample(self, rng, N):
        assert N <= len(self.values)
        result = rng.permutation(self.values)
        return result[:N]

    def widen(self, percent):
        raise NotImplementedError()

    def increase(self, percent):
        raise NotImplementedError()

@distribution
class MixtureDistribution(Distribution):
    def __init__(self, subdists, p=None):
        super().__init__(dict())
        self.dists = subdists
        self.p = p
        
    def _sample(self, rng, N):
        which = rng.choice(len(self.dists), size=N, p=self.p)
        result = np.zeros(N)
        for idx, dist in enumerate(self.dists):
            where = which == idx
            result[where] = dist(where.sum(), rng=rng)
        return result

    def widen(self, percent):
        for dist in self.dists:
            dist.widen(percent)

    def increase(self, percent):
        for dist in self.dists:
            dist.increase(percent)

@distribution
class NegateDistribution(Distribution):
    def __init__(self, subdist):
        super().__init__(dict())
        self.dist = subdist
        
    def _sample(self, rng, N):
        result = self.dist(N, rng=rng)
        return -result

    def widen(self, percent):
        self.dist.widen(percent)

    def increase(self, percent):
        self.dist.increase(-percent)

@distribution
class ScaleDistribution(Distribution):
    def __init__(self, subdist, scale):
        super().__init__(dict())
        self.dist = subdist
        self.scale = scale
        
    def _sample(self, rng, N):
        result = self.dist(N, rng=rng)
        return self.scale * result

    def widen(self, percent):
        self.dist.widen(percent)

    def increase(self, percent):
        self.dist.increase(-percent)

@distribution
class ExpDistribution(Distribution):
    def __init__(self, subdist):
        super().__init__(dict())
        self.dist = subdist
        
    def _sample(self, rng, N):
        result = self.dist(N, rng=rng)
        return np.exp(result)

    def widen(self, percent):
        self.dist.widen(percent)

    def increase(self, percent):
        self.dist.increase(-percent)

@distribution
class LogDistribution(Distribution):
    def __init__(self, subdist):
        super().__init__(dict())
        self.dist = subdist
        
    def _sample(self, rng, N):
        result = self.dist(N, rng=rng)
        return np.log(result)

    def widen(self, percent):
        self.dist.widen(percent)

    def increase(self, percent):
        self.dist.increase(-percent)


@distribution
class SumDistribution(Distribution):
    def __init__(self, subdists, weights=None):
        super().__init__(dict())
        self.dists = subdists
        if weights is None:
            weights = np.ones(len(subdists))
        self.weights = weights
        
        
    def _sample(self, rng, N):
        result = np.zeros(N)
        for weight, dist in zip(self.weights, self.dists):
            result +=  weight * dist(N, rng=rng)        
        return result

    def widen(self, percent):
        for dist in self.dists:
            dist.widen(percent)

    def increase(self, percent):
        for dist in self.dists:
            dist.increase(percent)

@distribution
class ProductDistribution(Distribution):
    """
    If the distributions only produce positive results, it is probably more
    efficient and numerically accurate to use sums of logs. However,
    if that is not possible, use this.
    Note that if exponents aren't integral, you may end up with complex results.
    """
    def __init__(self, subdists, exponents=None):
        super().__init__(dict())
        self.dists = subdists
        if exponents is None:
            exponents = np.ones(len(subdists))
        self.exponents = exponents
        
        
    def _sample(self, rng, N):
        result = np.ones(N)
        for exponent, dist in zip(self.exponents, self.dists):
            result +=  dist(N, rng=rng)**exponent
        return result

    def widen(self, percent):
        for dist in self.dists:
            dist.widen(percent)

    def increase(self, percent):
        for dist in self.dists:
            dist.increase(percent)

@distribution
class ChoiceDistribution(Distribution):
    """
    If the distributions only produce positive results, it is probably more
    efficient and numerically accurate to use sums of logs. However,
    if that is not possible, use this.
    Note that if exponents aren't integral, you may end up with complex results.
    """
    def __init__(self, choices, probs=None, replace=False):
        super().__init__(dict())
        self.choices = choices
        self.probs = None
        self.replace = replace
        if probs is not None:            
            if len(self.choices) != len(probs):
                raise ValueError("Choices and probs must have same length")
            if any(probs < 0):
                raise ValueError("All probs must be nonnegative")
            self.probs = probs / np.sum(probs)

    def _sample(self, rng, N):
        result = rng.choice(self.choices, p = self.probs, replace=self.replace, size=N)
        return result

    def widen(self, percent):
        raise NotImplementedError()

    def increase(self, percent):
        raise NotImplementedError()



ALIASES = dict(
    constant="ConstantDistribution",
    linear_ramp="LinearRampDistribution",
    uniform="UniformDistribution",
    normal="NormalDistribution",
    lognormal="LogNormalDistribution",
    lograndom="LogRandomDistribution",
    logbeta="LogBetaDistribution",
)

for alias in ALIASES:
    DISTRIBUTIONS[alias] = DISTRIBUTIONS[ALIASES[alias]]


PARAMS = [
    Parameter(symbol="A0",
              description="Modal (peak) abundance",
              def_min=10,
              def_max=100,
              def_mean=50,
              def_var=15,
              def_dist=LogRandomDistribution),
    Parameter(symbol="range",
              description="Range (width of distribution in environment)",
              def_min=50,
              def_max=150,
              def_mean=100,
              def_var=30,
              def_dist=NormalDistribution),
    Parameter(symbol="mode",
              description="Modal (peak) location",
              def_min=-95,
              def_max=195,
              def_mean=50,
              def_var=30,
              def_dist=UniformDistribution),
    Parameter(symbol="alpha",
              description="Peak shape (left skew)",
              def_min=2.5,
              def_max=6.5,
              def_mean=4,
              def_var=0.5,
              def_dist=UniformDistribution),
    Parameter(symbol="gamma",
              description="Peak shape (right skew)",
              def_min=2.5,
              def_max=6.5,
              def_mean=4,
              def_var=0.5,
              def_dist=UniformDistribution),
]


PARAMS = {p.symbol: p for p in PARAMS}


def gen_param_dist(param_name, param_dist=None, **kwargs):
    param = PARAMS[param_name]
    if param_dist is None:
        param_dist = param.def_dist
    return param_dist(defaults=param, **kwargs)


def force_range(dist: Distribution, shape: int|tuple[int],
                minval: float, maxval: float,
                rng: np.random.Generator):
    """Given a distribution, return array of 'shape' values drawn from
    the distribution, ensuring that all values lie within the given range.
    """
    result = dist(shape, rng=rng)
    while np.any(bad := ((result < minval) | (result > maxval))):
        if bad.sum() > result.size / 2:
            raise ValueError("Distribution lies too far out of "
                             f"[{minval}, {maxval}]")
        replace = dist(bad[bad].shape, rng=rng)
        result[bad] = replace
    return result


__all__ = [
    "RNG",
    "Distribution",
    "ConstantDistribution",
    "UniformDistribution",
    "NormalDistribution",
    "LogNormalDistribution",
    "LogRandomDistribution",
    "LinearRampDistribution",
    "force_range",
    "parse_distribution",
    ]
