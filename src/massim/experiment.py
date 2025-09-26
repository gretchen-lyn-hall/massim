from __future__ import annotations  # Allow forward declaration of types

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import defaultdict, namedtuple
from ctypes import ArgumentError
from typing import NamedTuple, DefaultDict, Iterable, TYPE_CHECKING, Any
import dataclasses as dc

from .distributions import RNG, Distribution, ConstantDistribution, DISTRIBUTIONS

import numpy as np
import pandas as pd

class Message:
    def __init__(self, target: str, name: str, _value=None, /,  **kwargs):
        """
        Class for passing messages to/from experiment stages.
        Messages are sent to the stage with name 'target'; the special
        target '__exec__' is reserved for sending messages to the execution
        system.
        For stage parameters, the default action for a message with the same
        name as a parameter is to set the parameter to the message payload.
        """
        self.target = target
        self.name = name
        if _value is None:
            self.value = kwargs
        else:
            self.value = _value
            if len(kwargs) > 0:
                raise ValueError("Message can have a single parameter "
                                 "or keyword parameters. Not both.")

@dc.dataclass(frozen=True)
class StageData:
    """Stage specific data"""
    
    _: dc.KW_ONLY
    abundance: np.ndarray|None = None
    # List of species group response objects; only used by core sim
    responses: list = dc.field(default_factory=list)
    # Raw beta response values, usable for probability
    # in noise routines (possibly after scaling by taking to a power)
    base_response: np.ndarray|None = None
    sample_coords: pd.DataFrame|None = None
    sample_info: pd.DataFrame|None = None
    species_info: pd.DataFrame|None = None

    def __post_init__(self):
        # To protect against programming errors, ensure that
        # the data is immutable.
        # Sadly, it's just too tricky to do this for pandas DataFrames.
        if self.abundance is not None:
            self.abundance.flags.writeable = False
        if self.base_response is not None:
            self.base_response.flags.writeable = False

    def copy(self, **kwargs):
        """Return a copy of the data, replacing any field with the values
        in kwargs."""
        return dc.replace(self, **kwargs)

    def has_fields(self) -> set[str]:
        return set([k for k, v in dc.asdict(self).items()
                    if v is not None])

                          
class ExperimentResult:
    def __init__(self,
                 data: StageData,
                 output_name: str,
                 output_index: dict[str, int]):
        self._output_name = output_name
        self._output_index = output_index
        self._data = data
        self._sub_results = {}

    @property
    def name(self) -> str:
        return self._output_name
    
    @property
    def full_id(self) -> str:
        tags = "_".join(f"{k}:{v}" for k, v in self._output_index.items())
        return self._output_name  + "_" + tags

    @property
    def short_id(self) -> str:
        tags = "_".join(f"{v}" for k, v in self._output_index.items())
        return self._output_name  + "_" + tags

    def get_index(self, key_or_pos: str|int):
        if isinstance(key_or_pos, int):
            if key_or_pos > len(self._output_index):
                raise KeyError("Output index out of range.")
            key_or_pos = list(self._output_index.keys())[key_or_pos]
        return self._output_index[key_or_pos]

    @property
    def run_index(self):
        return self._output_index

    @property
    def run_index_short(self):
        return list(self._output_index.values())

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data.abundance,
                            index=self._data.sample_info.index,
                            columns=self._data.species_info.index)
    @property
    def abundance(self) -> np.ndarray:
        return self._data.abundance
    
    @property
    def sample_info(self) -> pd.DataFrame:
        return self._data.sample_info

    @property
    def sample_coords(self) -> pd.DataFrame:
        return self._data.sample_coords

    @property
    def species_info(self) -> pd.DataFrame:
        return self._data.species_info
    
    
def _empty_np():
    return np.array([])

@dc.dataclass(frozen=True)
class PipelineData:
    data: StageData
    messages: list[Message]
    rng: np.random.Generator

    def copy(self, **kwargs):
        """Return a copy of the data, replacing any field with the values
        in kwargs."""
        return dc.replace(self, **kwargs)

    def target_messages(self, target: str, remove=False) -> list[Message]:
        result = [m for m in self.messages if m.target == target]
        if remove:
            self.messages[:] = [m for m in self.messages if m.target != target]
        return result

class StageParameter:
    class Instance:
        def __init__(self, parent: StageParameter,
                     name: str):
            self.parent = parent
            self.name = name

        def parse_message(self, payload):
            if self.parent.parser is None:
                if not isinstance(payload, self.parent.type_):
                    raise TypeError(f"Incorrect type for parameter '{self.name}'.")
                return payload
            else:
                return self.parent.parser(payload)
    
    def __init__(self, type_, default=None, msg_parser=None):
        self.type_ = type_
        self.default = default
        self.parser = msg_parser
        
def dist_parser(payload):
    if isinstance(payload, Distribution):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Message for distribution parameter must be a "
                        "Distribution or a dict.")
    dist_type = payload.get("dist_type")
    if dist_type is None:
        raise KeyError("Message for a distribution parameter must contain "
                       "a 'dist_type' entry.")
    if dist_type not in DISTRIBUTIONS:
        raise KeyError(f"Unknown distribution '{dist_type}' in message.")
        
    dist_cls, dist_args = DISTRIBUTIONS[dist_type]
    dist_args = [da.name for da in dist_args]
    kwargs = {}
    for dist_arg in dist_args:
        if not dist_arg in payload:
            raise KeyError(f"Distribution '{dist_type}' requires arguments: "
                           f"{', '.join(dist_args)}.")
        kwargs[dist_arg] = payload[dist_arg]

    return dist_cls(**kwargs)
        
        
class Stage(ABC):
    """Base class for a computational stage in an experiment.

    Each experiment is composed of a series of stages strung together (and
    possibly forking). A Stage provides one step of processing, and contains
    all the configuration and adjustable parameters for that step.
    The experimental pipeline takes care of handling data flow and updating
    of simple parameters (i.e. single values) in response to upstream messages
    such as parameter sweeps. 

    Subclasses must override the "default_name" and "execute" methods. The
    `execute` method is the crux of the stage; it is passed input (as
    StageData), an instance of "State" which contains all the modifiable
    parameters (e.g. anything that can be modified in a parameter sweep or
    other previous stage), and a random number generator. `execute` should
    then return a PipelineData instance, which contains the modified input
    (StageData), the RNG, and any messages for downstream stages.

    Subclasses should also define the class-level members
    REQUIRES and PROVIDES. Each is a list of strings; REQUIRES contains the
    names of the fields in StageData that this stage requires to compute its
    output, whereas PROVIDES contains the names of the fields that this
    stage creates or updates.
    
    If the stage needs to handle complex messages (other than setting simple
    parameters) it will also need to create an internal State class (subclass
    of Stage.State) that implements `handle_message` to parse and store the
    effects of the messages.

    """

    # Each subclass should redefine REQUIRES to include the fields in
    # StageData that are required to execute
    REQUIRES = []
    # Each subclass should redefine PROVIDES to include the fields in StageData
    # that it creates or alters
    PROVIDES = []

    class State:
        """Class for storing a stage's state in such a way that it can be
        updated via messages.
        The base state contains all parameters for the stage - simple variables
        that can be overridden by a message. If the state is more complex,
        subclass State to provide extra functionality (as well as overriding
        Stage.get_state)
        """
        def __init__(self, stage: Stage):
            self.stage = stage
            self.exec_info = {}
            for param in stage._params.values():
                setattr(self, param.name, getattr(stage, param.name))

        def __getitem__(self, key):
            return getattr(self, key)

        def handle_message(self, message: Message) -> bool:
            if message.name in self.stage._params:
                # todo: handle distributions or other types
                #
                param = self.stage._params[message.name]
                val = param.parse_message(message.value)
                setattr(self, param.name, val)
                return True
            elif message.name == "exec_info":
                # Special message containing information about the
                # current run
                self.exec_info = message.value
                return True
            return False

        def __repr__(self):
            out = ", ".join(f"{p.name} =  {getattr(self, p.name)}"
                            for p in self.stage._params.values())
            out += f", exec_info = {self.exec_info}";
            return f"StageState({out})"

        if TYPE_CHECKING:
            # Since State has dynamically generated members, we want
            # a way to hush the type checker.
            def __setattr__(self, name: str, value: Any, /) -> None:
                pass

            def __getattribute__(self, name: str, /) -> Any:
                pass

    def __new__(cls, *args, **kwargs):
        result = object.__new__(cls)
        params = {}
        for key, val in cls.__dict__.items():
            if isinstance(val, StageParameter):
                param = StageParameter.Instance(val, key)
                params[key] = param
        setattr(result, "_params", params)
        return result
    
    @abstractmethod
    def default_name(self) -> str:
        pass

    def __init__(self, stage_seed: int|list[int]|None = None, **kwargs):
        """
        If stage_seed is set, then all random computations for this stage
        will start at the same point. This essentially fixes the effect of
        this stage. This does not effect downstream random generation.
        """
        for param in self._params.values():
            if param.name in kwargs:
                pval = kwargs.pop(param.name)
                if not isinstance(pval, param.parent.type_):
                    # A special but common case is when a constant is used
                    # in place of a distribution. In that case, we'll replace
                    # it with a ConstantDistribution.
                    if param.parent.type_ == Distribution:
                        pval = ConstantDistribution(pval)
                    else:
                        raise TypeError(f"Parameter {param.name} must be of type "
                                        f"{param.parent.type_}.")
            else:
                pval = param.parent.default
            setattr(self, param.name, pval)
        if len(kwargs) > 0:
            unk = ", ".join(kwargs.keys())
            raise ValueError(f"Unknown kwargs: {unk}.")
        self.stage_seed = stage_seed

    def get_state(self) -> Stage.State:
        return self.State(self)

    def run(self,
            input: PipelineData
            ) -> PipelineData :

        has_fields = input.data.has_fields()
        missing = set(self.__class__.REQUIRES).difference(has_fields)
        if len(missing) > 0:
            raise ValueError(f"{self.default_name()} stage requires fields "
                             f"{', '.join(missing)} in input.")

        state = self.get_state()
        for message in input.messages:
            if not state.handle_message(message):
                raise ValueError("State received unknown message "
                                 f"'{message.name}.")

        if self.stage_seed is not None:
            rng = np.random.default_rng(self.stage_seed)
        else:
            rng = input.rng
            
        result = self.execute(input.data, rng, state)
        if self.stage_seed is not None:
            result = result.copy(rng=input.rng)
        return result

    @abstractmethod
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        pass

class ComboStage(Stage):
    """Utility class for creating a stage built from other stages.
    Some stages (such as the core simulator) can be split up into multiple
    substeps (e.g. generating species, generating sample coords, and applying
    them to yield an intensity matrix.) However, most use cases don't require
    that level of granularity, and could just use a single stage (e.g.
    a single core simulator stage).

    In that case, one can create a single combination stage class consisting of a
    linear pipeline of substages. ComboStage uses a little Python magic (i.e.
    introspection) to make it simple to generate such stage classes.

    To use it, subclass `ComboStage` and redefine the following class-level
    members:
      SUBSTAGES: A list (in order of execution) of (name, class) pairs.
        `name` should be a string that is a valid python identifier (no spaces,
         starts with alphabetic or underscore)
         'class' should be a subclass of Stage.
         The stages will all be instantiated and executed as a single unit
      DEFAULT_NAME: a string to use as the default name for this stage

    Example:
        class CoreSim(ComboStage):
            SUBSTAGES = [ ("spcs", GenSpeciesStage),
                          ("samp", GenSampleStage),
                          ("sim", ApplyCoreSimStage) ]
            DEFAULT_NAME = "core_sim"

        mysim = CoreSim(species_config=..., samplers=...)
        mysim.spcs  # Access the substages by name 


    Initializing:
    Any arguments used to initialize the substages can be passed to the
    combined stage, however all arguments must be passed as keyword arguments.
    E.g. If ComboABC is built from StageA, StageB, and StageC, and
    StageA's init method is defined as `StageA.__init__(self, arg_a)`,
    then to initialize ComboABC you must use `ComboABC(arg_a=<value>)`

    As with normal stages, you can initialize any substage parameters as
    keyword args as well, e.g. if param_b is a parameter for StageB and
    param_c1, param_c2 parameters for StageB:
      ComboABC(arg_a=<value>, param_b=3, paramc2=7)
    
    Limitations:
    Due to the fact that the stages share a single initialization, no two stages
    can share either an initialization argument name or parameter name. In particular,
    you can't include multiple copies of the same stage in a ComboStage. If
    there is an overlap, it will be detected the first time an instance of the
    ComboStage subclass is created, and an AttributeError will be raised.
    

    """
    SUBSTAGES = [] # list of name/class pairs
    DEFAULT_NAME = "Combo"
    
    def default_name(self):
        return self.__class__.DEFAULT_NAME

    def __init__(self, stage_seed: int|list[int]|None = None, **kwargs):
        import inspect
        sub_cls = self.__class__.SUBSTAGES
        sigs = [inspect.signature(cls.__init__)
                for _, cls in sub_cls]
        params = [inspect.getmembers(cls, lambda v: isinstance(v, StageParameter))
                  for _, cls in sub_cls]
        
        argmap = {}
        allargs = set()
        # Ensure that no two substages share init argument names
        for stage_idx, sig in enumerate(sigs):
            ps = [p for p in sig.parameters.keys()
                  if p not in ['self', 'args', 'kwargs']]
            clash = allargs.intersection(ps)
            if clash:
                raise AttributeError("Substages have overlapping init "
                                     f"arguments {clash}.")
            allargs.update(ps)
            for p in ps:
                argmap[p] = stage_idx
        # Do the same for stage parameters, as they can be passed in as kwargs
        for stage_idx, sub_params in enumerate(params):
            ps = [param_name for param_name, _ in sub_params]
            clash = allargs.intersection(ps)
            if clash:
                raise AttributeError("Substages have overlapping parameters "
                                     f"{clash}.")
            allargs.update(ps)
            for p in ps:
                argmap[p] = stage_idx
        print(allargs)

        super().__init__(stage_seed)

        # Prepare the argument lists for each substage
        init_args = [dict() for _ in sub_cls]
        for k, v in kwargs.items():
            idx = argmap.get(k)
            if idx is None:
                raise ArgumentError(f"Unknown argument '{k}'.")
            init_args[idx][k] = v

        # Finally, we can initialize out substages:
        self.stages = []
        for idx, (name, cls) in enumerate(sub_cls):
            new_stage = cls(**init_args[idx])
            self.stages.append(new_stage)
            setattr(self, name, new_stage)

    def run(self,
            input: PipelineData
            ) -> PipelineData :
        start_rng = input.rng
        if self.stage_seed is not None:
            input = input.copy(rng=np.random.default_rng(self.stage_seed))
        remaining_messages = input.messages
        states = []

        # We start by collecting the state for all stages and handling the
        # messages. This assumes that no two stages receive the same messages,
        # which probably should be verified
        for stage in self.stages:

            state = stage.get_state()
            remaining_messages = [msg for msg in remaining_messages
                                  if (not state.handle_message(msg)) or
                                  msg.target == "*"]
            states.append(state)

        if remaining_messages:
            raise ValueError("State received unknown message(s): "
                             f"{', '.join(m.name for m in remaining_messages)}.")

        # Then, we just execute the mini pipeline. This regurgitates much
        # of the mechanics of the base Stage.run method.
        for stage, state in zip(self.stages, states):
            has_fields = input.data.has_fields()
            missing = set(stage.__class__.REQUIRES).difference(has_fields)
            if len(missing) > 0:
                raise ValueError(f"{stage.default_name()} stage requires fields "
                                 f"{', '.join(missing)} in input.")

            
            input = stage.execute(input.data, input.rng, state)
        if self.stage_seed is not None:
            input = input.copy(rng=start_rng)
        return input


    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        # Only the substages `execute` methods get called.
        raise NotImplementedError("Should never get here")

        

                

class OutputFilter:
    """Base class for filtering output from an experiment.
    This class provides some basic functionality (removing species with
    presences below a certain threshold, normalizing the site/species matrix,
    etc) but can be overridden to provide more advanced functionality.
    """
    def __init__(self, min_abundance=0,
                 min_species_presence=0,
                 norm_method="none",
                 lognorm=False):
        self.min_abundance = min_abundance
        self.min_presence = min_species_presence
        self.norm_method = norm_method
        self.lognorm = lognorm

    def __call__(self, input: StageData) -> StageData:
        if input.abundance is None:
            return input
        abundance = input.abundance.copy()
        base_response = input.base_response.copy()
        thresh = abundance >= self.min_abundance
        abundance[~thresh] = 0
        presence = abundance > 0
        keep_spcs = presence.sum(axis=0) >= self.min_presence

        abundance = abundance[:, keep_spcs]
        base_response = base_response[:, keep_spcs]
        presence = presence[:, keep_spcs]
        species_info = input.species_info.iloc[keep_spcs]
        if self.lognorm:
            vals = abundance[presence]
            lambda_val = vals.min() / 10
            vals = np.log(vals + np.sqrt(vals**2 + lambda_val))
            abundance[presence] = vals

        if self.norm_method == 'sum':
            abundance /= abundance.sum(axis=1)[:, None]
        elif self.norm_method == 'none':
            pass
        else:
            raise ValueError("Unknown normalization method")
        return input.copy(abundance=abundance,
                          base_response=base_response,
                          species_info=species_info)
                          

        

class Experiment:
    """ Class for configuring and running a virtual simulation experiment.

    A virtual experiment consists of a set of simulation stages that are run in 
    sequence, with the output of each stage feeding to the input of the next.
    An experiment can contain replication or parameter sweep stages, which
    will lead to repeated execution of all downstream stages.
    The simplest experiments consist of a linear pipeline of stages, but
    it is possible to add forks to the pipeline (say, to apply two different
    noise treatments to simulation results).

    Ex:
    core_sim = GenCoreStage(...)
    rep1 = ReplicateStage(...)
    noise1 = QuantNoiseStage(...)
    rep2 = ReplicateStage(..)
    xfrms = TransformationStage(...)
    
    exp = Experiment("exp1")
    # Set the list of stages and name them 'cor', 'rep1, ...
    exp.set_stages(cor=core_sim, rep1=rep1, nois=noise1, rep2=rep2, xfrm=xfrms)

    By default, the last stage in any pipeline or fork will produce an output.
    However, outputs for any inner stage can be enabled:

    Ex:
    exp['nois'].enable_output(output_name="pre-transform")

    (The output_name is optional; by default, the outputs are named after
    their stage name)

    Running an experiement produces a sequence of results. For experiments
    with forks or multiple outputs, you can tell which stage it came from
    using ExperimentResult.name.

    Ex:
    for result in exp.run():
      if result.name == "pre-transform":
         ...

    You can determine which replicate (or step of a param sweep) a result
    comes from using the `run_index` method. This returns a dict, where the
    name of each repeating stage is a key, and the value is the step count.

    To just gather summary stats for an experiment, you can provide a user-
    defined function that takes an ExperimentResult and returns a dict or
    Pandas Series of statistics for that result. You can then pass the function
    to Experiment.gather, which will run the experiement and return a Pandas
    DataFrame of the statistics. The output dataframe will also contain columns
    for the output names and run indices.
   

    """

    class Node:
        def __init__(self,
                     experiment: Experiment,
                     stage: Stage,
                     name: str,
                     enable_output: bool | str = False):
            self.parent = experiment
            self.stage = stage
            self.name = name
            self.consumers: list[Experiment.Node] = []
            self.run_count = 0
            self.output_name = self.name
            self.output_result = enable_output
            if enable_output:
                if isinstance(enable_output, str):
                    self.output_name = enable_output

        def link_stage(self,
                       stage: Stage,
                       name: str | None = None):
            name = self.parent.check_name(stage, name)
            out_node = Experiment.Node(self.parent,
                                       stage,
                                       name)
            self.consumers.append(out_node)
            self.parent.all_nodes[name] = out_node
            return out_node

        def add_stages(self, *stages, **named_stages):
            """Add a chain of one or more stages after this node
            """
            tail = self

            for stage in stages:
                if tail is None:
                    tail = self.set_start_stage(stage)
                else:
                    tail = tail.link_stage(stage)
            for name, stage in named_stages.items():
                if tail is None:
                    tail = self.set_start_stage(stage, name=name)
                else:
                    tail = tail.link_stage(stage, name=name)
            return tail


        def enable_output(self,
                          output_name=None,
                          enable_output=True,
                          ) -> None:
            self.output_result = enable_output
            if output_name:
                self.output_name = output_name
                          

        def _prepare(self) -> None:
            self.run_count = 0

        def run(self,
                in_result: PipelineData,
                exec_count: int) -> PipelineData:
            self.run_count += 1
            node_msgs = in_result.target_messages(self.name)
            other_msgs = [m for m in in_result.messages if m.target != self.name]
            node_msgs.append(Message("*",
                                     "exec_info",
                                     exec_count=exec_count,
                                     first_run=self.run_count==1,
                                     node_name=self.name
                                     ))
            try:
                out_result = self.stage.run(in_result.copy(messages=node_msgs))
            except Exception as e:
                e.add_note(f"Encountered while in stage {self.name}")
                raise
            return out_result.copy(messages = other_msgs + out_result.messages)

    def __init__(self, name: str | None, output_filter: Callable|None = None):
        self.root: Experiment.Node | None = None
        self.name = name
        self.default_names = defaultdict(int)
        self.all_names: list[str] = []
        self.all_nodes: dict[str, Experiment.Node] = {}
        self.output_filter = output_filter

    def set_output_filter(self, output_filter: Callable|None):
        self.output_filter = output_filter

    def set_stages(self, *stages, **named_stages):
        """Link together a chain of experimental stages.

        This method is a handy method of creating an experiment that
        consists of a linear chain (no forks) of stages. Keyword arguments
        can be used to name the stages, otherwise the stage default names
        are used.

        If a fork is needed, you can set the main path of the experiment with
        this method, and then add forks using Node.add_stages.
        Ex: To add a fork after the third stage:
          exp = Experiment(s1=stage1, s2=stage2, s3=....)
          exp["s3"].add_stages(alt_stage1, alt_stage2)...

        Due to python constraints, you can't add unnamed stages after named
        stages.

        This method erases any stages already added to the experiment,
        and returns the node for the last stage in the chain.        
        """
        self.default_names = defaultdict(int)
        self.all_names: list[str] = []
        self.all_nodes: dict[str, Experiment.Node] = {}
        tail = None

        for stage in stages:
            if tail is None:
                tail = self.set_start_stage(stage)
            else:
                tail = tail.link_stage(stage)
        for name, stage in named_stages.items():
            if tail is None:
                tail = self.set_start_stage(stage, name=name)
            else:
                tail = tail.link_stage(stage, name=name)
        return tail
            



    def __getitem__(self, node_name):
        return self.all_nodes[node_name]

    def check_name(self, stage: Stage, name: str | None) -> str:
        if name is None:
            default_name = stage.default_name()
            self.default_names[default_name] += 1
            name = f"{default_name}_{self.default_names[default_name]}"
        if name in self.all_names:
            raise ValueError(f"Name '{name}' is already used in this experiment")
        return name

    def set_start_stage(self, stage: Stage, name: str|None = None) -> Experiment.Node:
        if self.root is not None:
            raise ValueError("Experiment already has start stage "
                             f"'{self.root.name}'.")
        name = self.check_name(stage, name)
        self.root = Experiment.Node(self, stage, name)
        self.all_nodes[name] = self.root
        return self.root

    class _ExecItem(NamedTuple):
        node: Experiment.Node
        node_input: PipelineData
        # For nodes that repeat (Replicate/ParamSweep), the number of
        # times they've been called
        node_count: int
        # Keep track of all repeating nodes, and track the count of each
        run_index: dict

    def _prepare(self):
        if self.root is None:
            raise ValueError("No stages have been added to experiment.")        
        stack: list[Experiment.Node] = [self.root]
        while stack:
            cur_node = stack.pop()
            cur_node._prepare()
            stack.extend(cur_node.consumers)

    def _repeating_stages(self):
        result = []
        if self.root is None:
            return []
        stack: list[Experiment.Node] = [self.root]
        while stack:
            cur_node = stack.pop()
            if isinstance(cur_node.stage, RepeatingStage):
                result.append(cur_node.name)
            cur_node._prepare()
            stack.extend(cur_node.consumers)
        return result
    
    def __repr__(self):
        if self.root is None:
            return f"[Experiment {self.name}: <empty>]"
        stack = []
        lines = [f"[Experiment {self.name}:"]
        def push_node(node, idx):
            if idx == 0:
                indent = " " * (4 * (1 + len(stack)))
                lines.append(f"{indent}{node.name}: {node.stage.__class__.__name__}")
            stack.append((node, idx))

        push_node(self.root, 0)
        while stack:
            node, cur_idx = stack.pop()
            if cur_idx < len(node.consumers):
                push_node(node, cur_idx + 1)
                push_node(node.consumers[cur_idx], 0)

        lines.append("]")
        return "\n".join(lines)
                
                 
        
    def run(self,
            messages=None,
            rng_or_seed:np.random.Generator|int|None = None,
            dry_run=False):
        """Run the experiment.
        Returns a generator that yields ExperimentResult items as outputs.
        All stages that are flagged as outputs (see 'link_stage') will be
        returned, as will any stage that has no children (that is, no ensuing
        stages).
        Each output has a "run index" that tracks which replicate it belongs to.
        The run_index is a Python dict: for for each repeating stage (such as
        a replicate or parameter sweep), the run_index contains the count
        for that stage. For example,  if there are three replicate stages
        named "R1", "R2", "R3", with each stage having 10 replicates, then
        the run index for the 209'th output will be:
          {"R1": 2, "R2": 0, "R3": 8}
        (the index is zero-based)
        """

        step_count = 0
        rng = RNG(rng_or_seed)
        # Essentially, this is just a depth-first traversal of the
        # experiment tree.
        if self.root is None:
            raise ValueError("No stages have been added to experiment.")
        self._prepare()
        
        exec_stack: list[Experiment._ExecItem] = []
        def push_exec(stage_node: Experiment.Node,
                      data: PipelineData,
                      count: int,
                      run_index: dict[str, int]):
            if isinstance(stage_node.stage, RepeatingStage):
                run_index = run_index.copy()
                run_index[stage_node.name] = count
            exec_stack.append(Experiment._ExecItem(stage_node,
                                                   data,
                                                   count,
                                                   run_index))

        if messages is None:
            messages = []
        elif not isinstance(messages, list):
            messages = [messages]
        for msg in messages:
            if msg.target not in self.all_nodes and msg.target != "*":
                raise ValueError(f"Message with unknown target "
                                 f"'{msg.target}'.")

        init_result = PipelineData(StageData(), messages=messages, rng = rng)

        # Prime the traversal with the root node and initial result.
        push_exec(self.root, init_result, 0, {})

        while exec_stack:
            step_count += 1
            # Run the next stage
            cur_node, cur_input, cur_count, run_idx = exec_stack.pop()
            if not dry_run or isinstance(cur_node.stage, RepeatingStage):
                next_result = cur_node.run(cur_input,
                                           cur_count)
            else:
                next_result = cur_input

            # Parse message results; either they are messages to the
            # run (target == "__exec__") or they are stored for downstream
            # use.
            exec_msgs = next_result.target_messages("__exec__", remove=True)
            for msg in exec_msgs:
                if msg.target == "__exec__":
                    if msg.name == "repeat":
                        if msg.value:
                            # For repeating stages, replace the node and its input
                            # on the stack
                            push_exec(cur_node, cur_input, cur_count+1, run_idx)
            # Check for messages with unknown targets
            for msg in next_result.messages:
                if msg.target not in self.all_nodes and msg.target != "*":
                    raise ValueError(f"Message with unknown target "
                                     f"'{msg.target}' generated by stage "
                                     f"'{cur_node.name}'.")

            if cur_node.output_result or len(cur_node.consumers) == 0:
                if dry_run:
                     yield run_idx
                else:
                    # Yield result
                    out_data = next_result.data
                    if self.output_filter:
                        out_data = self.output_filter(out_data)
                    out_result = ExperimentResult(out_data,
                                                cur_node.output_name,
                                                run_idx)
                    yield out_result
            for next_node in cur_node.consumers:
                push_exec(next_node, next_result, 0, run_idx)

    def count_iters(self, messages=None):
        return  len(list(self.run(messages=messages,
                                    rng_or_seed=None,
                                    dry_run=True)))
    def run_all(self,
            messages=None,
            rng_or_seed:np.random.Generator|int|None = None,
            ):
        """Run the experient,returning all the dataframes as output.
        If there is a single output stage, the results are returned as a list.
        Otherwise, the results are grouped by name.
        """
        from tqdm import tqdm
        
        num_out = len(list(self.run(messages=messages,
                                    rng_or_seed=None,
                                    dry_run=True)))
        results = []
        with tqdm(total=num_out) as progress_bar:
            for output in self.run(messages, rng_or_seed):
                results.append(output)
                progress_bar.update(1)

        names = set([x.name for x in results])
        if len(names) == 1:
            return results
        temp = defaultdict(list)
        for x in results:
            temp[x.name].append(x)

        # Create a object with fields for each output name.
        return NamedTuple("RunResult", [(k, list) for k in temp.keys()])(**temp)
            


    def gather(self, stat_func, messages=None, rng=None):
        """Run the experient, performing statistics on each output, and
        collect the results into a dataframe.
        'stat_func' must be a function taking an ExperimentResult as an
        input, and returning a type convertible to a Pandas Series (Series or
        a dict).
        The output dataframe will have statistics as columns, and will be
        indexed by the run indices of the output (see Experiment.run for a
        description of run indices).
        """
        from tqdm import tqdm
        
        num_out = len(list(self.run(messages=messages,
                                    rng_or_seed=None,
                                    dry_run=True)))
        index_keys = ["output_name"] + self._repeating_stages()
        results = []
        with tqdm(total=num_out) as progress_bar:
            for output in self.run(messages, rng):
                stats = stat_func(output)
                if isinstance(stats, pd.Series):
                    stats = stats.to_dict()
                stats.update(output.run_index, output_name=output.name)
                results.append(stats)
                progress_bar.update(1)
        result = pd.DataFrame.from_records(results)
        return result.set_index(index_keys)
        

class RepeatingStage(Stage):
    """ Base class for stages that repeat, like replication or parameter
    sweeps.
    """
    def __init__(self, freeze_rng: bool=False, **kwargs):
        """
        If 'freeze_rng' is True, then the random number generator is reset
        to the same state for all downstream stages.
        """
        super().__init__(**kwargs)
        self.freeze_rng = freeze_rng
        self._rng_state = None

    @abstractmethod
    def is_done(self, state: Stage.State) -> bool:
        pass


    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        if self.freeze_rng:
            if state.exec_info["first_run"]:
                self._rng_state = rng.bit_generator.state
            else:
                assert self._rng_state is not None
                rng.bit_generator.state = self._rng_state
        msgs = []
        msgs.append(Message("__exec__", "repeat", not self.is_done(state)))
                
        # do stuff
        result = PipelineData(input, msgs, rng)

        return result
            
            

    
class ReplicateStage(RepeatingStage):
    replicates = StageParameter(int, 1)
    
    def default_name(self) -> str:
        return "Replication"

    def is_done(self, state: Stage.State) -> bool:
        # exec_count starts at 0 for first run, so the last run is
        # 'replicates-1'.
        return state.exec_info["exec_count"] >= state.replicates - 1

    # Uses parent's execute method

class SweepVals:
    @abstractmethod
    def get_value(self, step: int, num_steps: int, rng: np.random.Generator):
        pass

    @property
    @abstractmethod
    def fixed_steps(self) -> None|int:
        pass
    
class LinSpace(SweepVals):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_value(self, step, num_steps, rng):
        return self.start + (self.end - self.start) * step/(num_steps - 1)

    @property
    def fixed_steps(self):
        return None

class ArraySweep(SweepVals):
    def __init__(self, values: Iterable):
        self.values = list(values)

    def get_value(self, step, num_steps, rng):
        if step >= len(self.values):
            raise RuntimeError("Sweep step out of bounds")
        return self.values[step]

    @property
    def fixed_steps(self):
        return len(self.values)

class RandomSweep(SweepVals):
    def __init__(self, dist):
        self.dist = dist

    def get_value(self, step, num_steps, rng):
        return self.dist(rng=rng)

    @property
    def fixed_steps(self):
        return None

    
class ParameterSweep(RepeatingStage):
    num_steps = StageParameter(int, 10)

    class Sweep(NamedTuple):
        target: str
        name: str
        range: (SweepVals | dict)

        def make_message(self,
                         cur_step: int,
                         num_steps: int,
                         rng: np.random.Generator) -> Message:
            if isinstance(self.range, SweepVals):
                return Message(self.target,
                               self.name,
                               self.range.get_value(cur_step, num_steps, rng))
            else:
                out_dict = dict()
                for k, v in self.range.items():
                    if isinstance(v, SweepVals):
                        out_dict[k] = v.get_value(cur_step, num_steps, rng)
                    else:
                        out_dict[k] = v
                return Message(self.target, self.name, out_dict)
                    
            
    
    def default_name(self) -> str:
        return "ParameterSweep"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = []
        # Track if any of the sweeps has a fixed number of steps.
        # If so, the num_steps parameter can't be overridden, and
        # all fixed sweeps must have the same number of steps
        self.fixed_steps = None

    def _check_sweep(self, sweep_vals: SweepVals) -> None:
        if sweep_vals.fixed_steps is None:
            return
        if self.fixed_steps is not None:
            if sweep_vals.fixed_steps != self.fixed_steps:
                raise ValueError("All fixed outputs in a sweep must "
                                 "have the same number of values.")
        else:
            self.fixed_steps = sweep_vals.fixed_steps
            self.num_steps = self.fixed_steps
            
    def add_sweep(self, target: str, name: str, val_range) -> ParameterSweep:
        if isinstance(val_range, SweepVals):
            self._check_sweep(val_range)
            self.outputs.append(ParameterSweep.Sweep(target, name, val_range))
        elif isinstance(val_range, dict): 
            val_dict = val_range.copy()
            seen = False
            for val in val_range.values():
                if isinstance(val, SweepVals):
                    seen = True
                    self._check_sweep(val)
            self.outputs.append(ParameterSweep.Sweep(target, name, val_dict))
        else:
            raise ValueError("Sweep value must be a sweep value object or "
                             "a dict containing at least one sweep value")
        return self

    def is_done(self, state: Stage.State) -> bool:
        return state.exec_info["exec_count"] >= state.num_steps - 1
    
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        result = super().execute(input, rng, state)
        cur_step = state.exec_info["exec_count"]
        for output in self.outputs:
            result.messages.append(output.make_message(cur_step, state.num_steps, rng))

        return result
    

class DebugStage(Stage):
    test_val = StageParameter(float, 10)

    def default_name(self) -> str:
        return "Debug"

    def __init__(self, txt, **kwargs):
        super().__init__(**kwargs)
        self.text = txt

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        print("InExecute: ", self.text, state.exec_info['node_name'], state.test_val)
        print("    Random:", rng.uniform(0, 100))
        result = PipelineData(input, [], rng)
        return result


class Debug2Stage(DebugStage):
    test_val2 = StageParameter(float, 10)

    def default_name(self) -> str:
        return "Debug2"

    def __init__(self, txt, **kwargs):
        super().__init__(txt, **kwargs)
        self.text = txt

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        print("InExecute: ", self.text, state.exec_info['node_name'], state.test_val)
        print("    Random:", rng.uniform(0, 100))
        result = PipelineData(input, [], rng)
        return result
