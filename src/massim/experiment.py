"""Define and run MASSIM simulation experiments.

Basic Concepts:
---------------

A MASSIM "Experiment" is a way to define ensembles of simulations, with each
simulation built up as a pipeline of relatively simple stages. Each stage acts
as a processing step that can modify data from the stages before it. An
Experiment can be as simple as generating a single output from a single stage,
or it can define parameter sweeps, replicates, and multiple treatments.

When an Experiment is run, it generates an ensemble of ExperimentResults. 
Each ExperimentResult typically contains a complete synthetic FT-ICR dataset,
consisting of an intensity matrix, metabolite mass and metadata, and
sampling location and metadata.
(There are some exceptions to this, depending on how the Experiment
is set up; an ExperimentResult can contain results from intermediate stages
such as species abundance. However, the typical expectation is that the output
is a mass spec simulation).

As stated, an Experiment is built up as a pipeline of simpler stages. These
stages can roughly be broken down into generators (create new data from
simulation parameters), data transformers (modify data from previous stages)
and control (create replicates and parameter sweeps). Each stage feeds its
output forward to the next stage in the pipeline. For repeating stages (
such as replicates and parameter sweeps), the input data is saved and
all downstream (following) stages will be repeated with the same input.

As an example, a pipeline might have the structure (Note - this is pseudocode;
for details on how to set up experimenta and parameter sweeps, consult the
class documentation.)

ParameterSweepStage(species_turnover=(1 to 5), n=10)
 -> COMPASSimStage
   -> ReplicateStage(replicates=5)
      -> GaussianNoiseStage

(Here, COMPASSimStage uses the COMPAS ecological simulation algorithm to
simulate species abundance across a virtual landscape. It doesn't produce
FT-ICR spectra directly, but it is often used in tandem with other stages.)

The above experiment would produce 50 outputs. It would vary the
`species_turnover` parameter from 1 to 5 in 10 steps. For each value,
it would create and run a COMPAS simulation to produce a species-abundance
output. For each COMPAS output, the output is replicated 5 times, and gaussian
noise is applied.

Pipelines can be more complex than the above; they can fork (to output, say,
different noise treatments), and intermediate results can be returned as well
(if you want to compare the raw COMPAS output to the noised output)

The same experiment can be run multiple times. Since most simulation stages use
a random number generator, the results will be different each run (unless
you explicitly seed the RNG).


Experiment Outputs:
-------------------------

Experiments are defined as pipelines of stages. By default, the outputs of
an Experiment will be the outputs of all terminal stages (that is, stages that
don't act as inputs to other stages). However, intermediate results can be
returned as well.

Consider the following experiment

exp = Experiment("test")
exp.set_stages(my_stage_1=...
               rep_1=ReplicateStage(n_replicates=5),
               my_stage_2=...
               rep_2=ReplicateStage(n_replicates=10),
               my_final_stage=...)

Here, I've named the stages 'my_stage_1', 'rep_1', etc., but you can choose
whatever names you desire.

By default, running the experiment will only output results from
`my_final_stage`. We can enable output from other stages with:

exp["my_stage_2"].enable_output()

Now, when the experiment is run, outputs from both `my_stage_2` and
`my_final_stage will be output.

Each output will contain the following properties:
.df: Pandas dataframe containing the sample x metabolite (or species)  matrix
.abundance: 2D numpy array of abundances; same values as .df
.sample_info: Information for each sample (rows of .df), including gradient
   locations.
.species_info: Information for each metabolite/species (columns of .df).
   This includes the species group, and for metabolites the metabolite mass.
.name: Name of the stage that produced this output
.run_index: When repeating stages are used, the iteration index for each
  stage

There are more fields; see ExperimentResult docs for details.

.base_response: 2D numpy array representing the relative presence (0-1) of
   each metabolite/species. Used for presence/absence noise computation



Disambiguating outputs:
-----------------------

When multiple outputs are enabled, or parameter sweeps are used, it may
be necessary to know which ExperimentResult came from which stage, and
from which parameter sweep value.

An ExperimentResult has two properties that can be used.
`ExperimentResult.name` is the name of the stage that produced the result.
In the example above, the result name would be either `my_stage_2` or
`my_final_result`. You can use this to conditionally process results:

for result in exp.run():
  if result.name == "my_final_stage":
     # process final stage
  else:
     # process intermediate.

To determine which replicate and/or which step of the parameter sweep the
result is from, you can use ExperimentResult.run_index. This returns a
dictionary with the name of all repeating stages, and the 0-based index.
For the example above, the run_index would be a dict
{"rep_1": <int from 0 to 4>,
 "rep_2": <int from 0 to 10>}

For parameter sweeps, `run_index` only returns the index, not the parameter
value(s) of the sweep. There is a way to get at the actual values, but
currently it's a bit ugly.


Running Experiments:
--------------------

Experiments can be run in different ways. The simplest method is to
run everything at once, with `exp.run_all()`, which returns a list of
results. However, as each individual output is represented as a large
metabolite x sample intensity matrix, this approach can consume vast
amounts of memory.

Instead, there are more memory-efficient ways of running an experiment.
Bear with me, as I've experimented with multiple methods, and the
Experiment class is a bit cluttered with my own, er, experiments.

First, it can be run as a generator:

for result in exp.run():
  # process results

Second, if you just want to run some summary statistics on each result, you
can use `Experiment.map_mp` or `Experiment.map_sp`. The two are functionally
equivalent, but `map_mp` uses multiprocessing to run the experiment in parallel
across multiple CPU cores. This can greatly speed up lengthy experiments, but
may not work on all platforms or with all stages.

The map_* functions take a function as an input. They then run the experiment,
applying the function to each enabled output stage, and store the function result.
The output of the map_* functions is a list of tuples containing:
- the output's stage name
- the output's run index
- the function result.

Similarly, there is the "gather" method. It behaves like the map_* functions,
except that it expects the function to return a (flat) dictionary of values
containing summary statistics for each output.
The summary statistics are then arranged in a DataFrame. Each value in the
dictionary gets its own column. In addition, the output contains columns
for:
- the output name
- each run index (that is, one column per repeating stage in the experiment)

Messages:
---------

This is more of an implementation detail, but this is how stages can communicate
with the Experiment or each other.
For instance, the ParameterSweep stage uses messages to inform downstream
stages to modify their parameters, and RepeatingStages use messages to inform
the Experiment when they need to be repeated, or when all repetitions are
finished.

A message consists of three parts:
- `target`: the name of the stage that should receive this message. The special
   name "__exec__" is used for messages to the Experiment.
- `name`: really, should be called `command`!
  - For messages to stages, this can be the name of the parameter to alter
    (during a parameter sweep, say)
  - The special name "exec_info" is reserved for messages from the Experiment
    to a stage
- `value`: the payload of the message. For parameter change messages, this
  is the new value of the parameter. Other messages have specified payload
  types.

Stage State:
------------

More implementation details:

Running a complex pipeline with potential
forks, multiple repetitions, changing parameters and multiple outputs
requires a moderately sophisticated way of handling state. To ensure that a
stage is run with exactly the parameters it should have, we a State class
to encapsulate all stage parameters.

When a stage is defined and added to an experiment, the user can set various
parameters for that stage (e.g. number of species and species turnover in
a COMPAS stage). Most stage parameters can also be controlled through messages.
This allows them to be varied via parameter sweeps, or overridden in a particular
experiment run.

A particular stage may be executed many times in the course of an experiment
run. When a stage is executed, it first copies all of its user defined
parameters into a State object. It then processes any parameter change messages.
These parameter changes only apply to the State object, ensuring that the
original user defined parameters are untouched. As it executes, the Stage
takes any parameter values from the State object, which ensures that it
always has the correct, up to date parameter values.

For simple parameters (scalar values or distributions), the default State
class already has an implementation to handle the copying and message
passing. For more complex parameters (or for stages like GenSpeciesStage
and GenProfileStage that hold multiple, complex configurations), the
stage will need to define its own State class with its own message handling
method.


"""
from __future__ import annotations  # Allow forward declaration of types

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import defaultdict, namedtuple
from ctypes import ArgumentError
from typing import NamedTuple, DefaultDict, Iterable, TYPE_CHECKING, Any, Dict, List
import dataclasses as dc
import pdb
import logging
from queue import Empty, Full

from .distributions import RNG, Distribution, ConstantDistribution, DISTRIBUTIONS

import numpy as np
import pandas as pd

CEASE_VALUE = "CEASE"

class MockJoinableQueue:
    def __init__(self):
        self.queue = []
        self.task_count = 0

    def put(self, item):
        self.queue.append(item)
        self.task_count += 1

    def get(self):
        pass

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
    """Stage specific data, used to pass data between stages

    In general, this should not be used in user code.
    It is a dataclass that reserves spots for abundance matrices,
    species and sample info, and sample coordinates.
    For stages that produce other types of data, they can store that in
    the `extra` field.

    Note, this class automatically marks the abundance and base_response
    fields as *read-only* numpy arrays. This is to prevent one stage from
    inadvertently modifying its inputs, which could have unintended consequences.

    Stages should always produce output by using the `copy` method to copy their
    input; any fields that the stage modifies can be replaced using the keyword
    args. This ensures that a stage passes through any fields that it does not
    use, but might be needed by downstream stages.
        
    """
    
    _: dc.KW_ONLY
    abundance: np.ndarray | None = None
    # List of species group response objects; only used by core sim
    responses: list = dc.field(default_factory=list)
    # Raw beta response values, usable for probability
    # in noise routines (possibly after scaling by taking to a power)
    base_response: np.ndarray | None = None
    sample_coords: pd.DataFrame | None = None
    sample_info: pd.DataFrame | None = None
    species_info: pd.DataFrame | None = None
    extra: Dict = dc.field(default_factory=dict)

    def __post_init__(self):
        # To protect against programming errors, ensure that
        # the data is immutable.
        # Sadly, it's just too tricky to do this for pandas DataFrames.
        if self.abundance is not None:
            self.abundance.flags.writeable = False
        if self.base_response is not None:
            self.base_response.flags.writeable = False
        # Make sure no extra fields conflct with base fields
        assert len(set(self.__dataclass_fields__).intersection(self.extra.keys())) == 0

    def copy(self, **kwargs):
        """Return a copy of the data, replacing any field with the values
        in kwargs."""
        # We have to take special care to explicitly copy the "extra" field,
        # updating it with any new values
        new_extra = self.extra.copy()
        new_extra.update(kwargs.pop("extra", {}))
        result = dc.replace(self, extra=new_extra, **kwargs)
        return result

    def has_fields(self) -> set[str]:
        return set([k for k, v in dc.asdict(self).items()
                    if v is not None and k != "extra"]).union(self.extra.keys())

                          
class ExperimentResult:
    """
    Class to contain a single result from a MASSIM experiment.
    
    While a result can theoretically contain any kind of data, this class
    is geared towards results that can be expressed as a sample x metabolite
    or sample x species matrix. Methods are provided to access these matrices,
    variants of these matrices, and sample and metabolite metadata.
    """
    def __init__(self,
                 data: StageData,
                 output_name: str,
                 output_index: dict[str, int],
                 messages: dict = None,
                 states: dict = None):
        """
        Constructor. Should generally not be called from user code, as it is
        populated by Experiment.
        """
        self._output_name = output_name
        self._output_index = output_index
        self._data = data
        self._sub_results = {}
        if messages is None:
            messages = []
        self._messages = messages
        if states is None:
            states = {}
        self._states = states

        
    @property
    def name(self) -> str:
        """
        Name of the Experiment stage that generated this result.
        """
        return self._output_name

    @property
    def run_index(self):
        """
        For MASSIM experiments with repeating stages (ReplicateStage or
        ParameterSweep), indicates which repetition produced this result.
        Returns a dictionary of <repeating stage name>: <repetition index>.
        """
        return self._output_index

    
    @property
    def full_id(self) -> str:
        """
        String that combines the full name and run index of this result.
        Of the form
        "<stage name>_<1st rep stage>_<1st rep index>_<2nd rep stage>_..."
        """
        tags = "_".join(f"{k}:{v}" for k, v in self._output_index.items())
        return self._output_name + "_" + tags

    @property
    def short_id(self) -> str:
        """
        A shorter version of full_id that contains only the name and run
        indices, and omits the repeating stage names.
        """
        tags = "_".join(f"{v}" for k, v in self._output_index.items())
        return self._output_name + "_" + tags

    def get_index(self, key_or_pos: str | int):
        """
        Get a particular entry in the run index, either by position (if
        `key_or_pos` is an integer) or by name.
        """
        if isinstance(key_or_pos, int):
            if key_or_pos > len(self._output_index):
                raise KeyError("Output index out of range.")
            key_or_pos = list(self._output_index.keys())[key_or_pos]
        return self._output_index[key_or_pos]


    @property
    def run_index_short(self):
        """
        Like run_index, but only returns a tuple of run indices, and not
        the repeating stage names.
        """
        return list(self._output_index.values())

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the sample x metabolite (or sample x species) abundance matrix
        of the output, expressed as a Pandas dataframe.
        The rows are indexed identically to this result's `sample_info`,
        and the columns are indexed identically to this result's `species_info`.
        """
        return pd.DataFrame(self._data.abundance,
                            index=self._data.sample_info.index,
                            columns=self._data.species_info.index)

    @property
    def dfmass(self) -> pd.DataFrame:
        """
        Return the sample x metabolite abundance matrix
        of the output, expressed as a Pandas dataframe.
        Like the `df` property, but columns are indexed by metabolite mass,
        rather than by a simple range index.
        """
        return pd.DataFrame(self._data.abundance,
                            index=self._data.sample_info.index,
                            columns=self._data.species_info.mass)

    @property
    def abundance(self) -> np.ndarray:
        """
        Return the sample x metabolite abundance matrix as a 2D numpy array.
        """
        return self._data.abundance

    @property
    def presence(self) -> np.ndarray:
        """
        Return sample x metabolite/species presence boolean presence matrix
        as a 2D numpy array. Here, presence is defined as strictly positive
        entries in the abundance matrix.
        """
        return self._data.abundance > 0
    
    @property
    def sample_info(self) -> pd.DataFrame:
        """Return metatada about the samples, expressed as a Pandas
        DataFrame.
        Typically contains a generated sample name, as
        well as sampling locations (gradient coordinates).

        """
        return self._data.sample_info

    @property
    def sample_coords(self) -> pd.DataFrame:
        """
        Return sample coordinates as a Pandas DataFrame.
        Typically, sampling coordinates are gradient locations in a
        virtual landscape, with each gradient spanning from 0-100.
        Columns are indexed by gradient name, and rows by sample id.
        """
        return self._data.sample_coords

    @property
    def species_info(self) -> pd.DataFrame:
        """Return metadata about abundance matrix columns.

        Here 'species' is used as a general term; for the output of
        COMPAS simulations, it refers to biological species, whereas
        for metabolite outputs it refers to chemical species.
        For metabolite outputs, this dataframe contains a 'mass' column,
        which represents neutral metabolite mass.

        """
        return self._data.species_info
    
    
def _empty_np():
    return np.array([])

@dc.dataclass(frozen=True)
class PipelineData:
    """
    A wrapper class for passing data between stages and the Experiment.

    In general, should not be accessed by user code.
    
    While the StageData class contains the numerical simulation data to be
    passed between stages, the PipelineData contains additional data necessary
    for managing the pipeline execution.

    In particular it contains:
    - `messages`: a list of messages to pass to and from a stage and the
       executor (the Experiment).  Messages can also be passed from one stage
       to downsteam stages.
    - `rng`: the current state of the random number generator. Some states
       may work by `freezing` the rng at a particular state; this method allows
       them to pass the frozen state downstream.
    - `states`: a history of Stage states for all *upstream* (prior) stages.
       This is used to track the provenance of every output; i.e. so that
       we can associate
    
    """
    data: StageData
    messages: list[Message]
    rng: np.random.Generator
    states: dict = dc.field(default_factory=dict)

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
    """
    Class that can be used within a Stage to specify a user-configurable
    parameter.

    This is a convenience for supplying stages with configurable parameters,
    with a whole bunch of mechanisms to automatically handle common cases.
    
    In particular, if a Stage defines a StageParameter (by specifying it as
    a *class* member), then the following functionality is enabled:
    - Automatic initialization of the parameter from constructor keyword
      arguments. E.g., something like: MyState(my_param=5)
    - All StageParameters are automatically copied to the Stage's State class
      upon every stage execution.
    - Automatic message handling: Any message with the same name as the
      StageParameter will be used to update the parameter's value in the
      State instance.


    Usage:
    class MyStage(Stage):
       my_param = StageParameter(<type>, <default_value>, [<optional message parser>])
       my_int_param = StageParameter(int, 5)
       my_dist_param = StageParameter(Distribution, NormalDistribution(0,1),
                                      msg_parser=dist_parser)

       def __init__(self, **kwargs):
          super().__init__(**kwargs)
          # handles `MyStage(my_int_param=7, my_dist_param=UniformDistribution(0, 1))          
    """
    class Instance:
        """
        Generates an instance member from StageParameter
        """
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
    """
    Message parser for distribution objects.
    Can parse messages where the message value is:
    - A Distribution instance (just copies the distribution)
    - A dict, with the key "dist_type" for the name of the distribution,
      and keys for each of the distribution parameters.
      E.g. {"dist_type": "normal", "mean": 0, "std": 1}
    """
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
            """
            Default Stage.State constructor.
            Copies all its parent stage's StageParameter values into itself.
            """
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
        """
        Python magic mumbo jumbo to handle StageParameters and convert them
        into object members.
        """
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
        """
        This is a bit overkill, since the Experiment provides such an
        easy way to specify names.
        However, if a name *isn't* provided when defining an experiment,
        this method is called to generate a base name for the stage.
        Multiple stages will have names based off `default_name`
        (e.g. `DefaultName1`, `DefaultName2`, etc.
        """
        pass

    def __init__(self, stage_seed: int|list[int]|None = None, **kwargs):
        """
        Constructor that handles the base Stage construction. Must be
        called by all subclasses.
        
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
            input: PipelineData,
            as_name: str,
            debug=False,
            ) -> PipelineData:
        """
        Handle the housekeeping associated with running this stage.

        Most Stage subclasses will *not* need to override this method;
        they should only override `execute`.
        
        This method is responsible for ensuring the stage input has
        all the REQUIRED fields, generating the Stage.State for this
        execution, parsing messages,  handling RNG freezing, and running
        the stage's `execute` method.

        This is called with the input PipelineData, as well as this stage's
        name in the experiment. Optionally, if 'debug' is set,
        the stage's `execute` method will be run in the debugger.
        """
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
        if debug:
            result = pdb.runcall(self.execute, input.data, rng, state)
        else:
            result = self.execute(input.data, rng, state)
        out_states = input.states.copy()
        out_states[as_name] = state
        result = result.copy(states=out_states)
        if self.stage_seed is not None:
            result = result.copy(rng=input.rng)
        return result

    @abstractmethod
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        """
        Abstract data processing method. MUST be overridden by subclasses.
        """
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
    SUBSTAGES = []  # list of name/class pairs
    DEFAULT_NAME = "Combo"
    
    def default_name(self):
        return self.__class__.DEFAULT_NAME

    def __init__(self, stage_seed: int | list[int] | None = None, **kwargs):
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
            input: PipelineData,
            as_name: str,
            debug=False
            ) -> PipelineData:
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
        out_states = input.states.copy()
        for stage, state in zip(self.stages, states):
            out_states[as_name + "." + stage.default_name()] = state
            has_fields = input.data.has_fields()
            missing = set(stage.__class__.REQUIRES).difference(has_fields)
            if len(missing) > 0:
                raise ValueError(f"{stage.default_name()} stage requires fields "
                                 f"{', '.join(missing)} in input.")

            
            if debug:
                input = pdb.runcall(stage.execute, input.data, input.rng, state)
            else:
                input = stage.execute(input.data, input.rng, state)
        if self.stage_seed is not None:
            input = input.copy(rng=start_rng)
        
        return input

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        # We overrode the 'run` method to directly call the substages'
        # execute methods.
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
    exp['nois'].enable_output(True)

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
        """
        The Node class is used to wrap each experiment Stage. It contains all
        the information necessary to run the Stage as part of an experiment
        pipeline; the stage's name in the experiment, its position in the
        pipeline, whether the stage should be output, etc.

        User code should not usually create or run nodes directly.
        The Experiment exposes the nodes for the purposes of using the
        `link_stage`, `add_stages`, and `enable_output` methods.
        """
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
            self.debug = False
            self.output_filter = None
            if enable_output:
                if isinstance(enable_output, str):
                    self.output_name = enable_output

        def link_stage(self,
                       stage: Stage,
                       name: str | None = None):
            """
            Add the given `stage` to this experiment, to follow directly
            after this node's stage.

            This method can be used to create a forking pipeline. The
            output from this stage can be fed into an arbitrary number
            of downstream stages.

            Usage:
            exp = Experiment("test")
            exp.set_stages(stage1=..., stage2=..., stage3=...)
            # Fork stage 2 to also output into a new stage:
            exp["stage_2"].link_stage(SomeStage(), name="stage_3_alt")
            """
            name = self.parent._check_name(stage, name)
            out_node = Experiment.Node(self.parent,
                                       stage,
                                       name)
            self.consumers.append(out_node)
            self.parent.all_nodes[name] = out_node
            return out_node

        def add_stages(self, *stages, **named_stages):
            """Add a chain of one or more stages after this node

            This method can be used to add a fork to a pipeline. Output
            from this stage will be sent into the sub_pipeline specified
            in the arguments.

            Usage:
            exp = Experiment("test")
            exp.set_stages(stage1=..., stage2=..., stage3=...)

            # Fork stage 2 to also output into stage4->stage5
            exp["stage_2"].add_stages(stage4=SomeStage(),
                                      stage5=SomeOtherStage())

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
                          val: bool = True,
                          output_name: str | None = None,
                          output_filter: Callable | None = None
                          ) -> None:
            """
            Request that this stage output its results, even if it is
            an intermediate stage.

            Output can be enabled or disabled by passing a bool (default
            True).
            The name of the output's ExperimentResult can be defined; by
            default it is equal to the stage's name in the experiment.

            Optionally, a filter can be applied to the output. This filter
            can be any callable, or an OutputFilter instance. The purpose
            is to perform any common preprocessing (intensity thresholding,
            normalization, etc) on the outputs.
            """
            if val is None:
                val = True
            if isinstance(val, bool):
                self.output_result = val
            if self.output_result:
                if output_name is not None:
                    self.output_name = output_name
                self.output_filter = output_filter
                
        def _prepare(self) -> None:
            """
            Prepare this node at the beginning of an experiment run.
            """
            self.run_count = 0

        def _run(self,
                 in_result: PipelineData,
                 exec_count: int) -> PipelineData:
            self.run_count += 1
            node_msgs = in_result.target_messages(self.name)
            other_msgs = [m for m in in_result.messages if m.target != self.name]
            node_msgs.append(Message("*",
                                     "exec_info",
                                     exec_count=exec_count,
                                     first_run=self.run_count == 1,
                                     node_name=self.name
                                     ))
            try:
                out_result = self.stage.run(in_result.copy(messages=node_msgs),
                                            self.name,
                                            debug=self.debug)
            except Exception as e:
                e.add_note(f"Encountered while in stage {self.name}")
                raise
            return out_result.copy(messages=node_msgs + other_msgs + out_result.messages)

        def _build_out_tree(self, tree: Experiment.OutTree) -> None:
            for child in self.consumers:
                tree.add_node(self.name, child.name)
                child._build_out_tree(tree)

    class OutTreeNode:
        """
        Ignore for now. Supposed to be a nice way of arranging outputs,
        but unfinished.
        """
        def __init__(self, name: str):
            self._name = name
            self._children: dict[str, Experiment.OutTreeNode] = {}
            self._outputs: list[ExperimentResult] = []

        def __getitem__(self, idx: int) -> ExperimentResult:
            return self._outputs[idx]

        def _add_child(self, name: str) -> Experiment.OutTreeNode:
            if name in self._children:
                raise ValueError(f"Child '{name}' already exists.")
            result = Experiment.OutTreeNode(name)
            self._children[name] = result
            setattr(self, name, result)
            return result

    class OutTree:
        def __init__(self, root_name: str):
            self.root = Experiment.OutTreeNode(root_name)
            self.nodes = {root_name: self.root}
            pass

        def add_node(self, from_node: str, to_node: str):
            assert from_node in self.nodes
            assert to_node not in self.nodes
            node = self.nodes[from_node]
            self.nodes[to_node] = node._add_child(to_node)

        def add_output(self,
                       result: ExperimentResult):
            assert result.name in self.nodes
            self.nodes[result.name]._outputs.append(result)

        def prune(self):
            from types import SimpleNamespace
            result = Experiment.OutTreeNode("")

            


            def prune_node(out_node, at_node):
                next_out = out_node
                if len(at_node._outputs) > 0:
                    # Add a level to the result
                    out_child = out_node._add_child(at_node._name)
                    out_child._outputs = at_node._outputs
                    next_out = out_child
                if at_node._children:
                    for child_name, child in at_node._children.items():
                        prune_node(next_out, child)
                            
            prune_node(result, self.root)
            return result
            

    def __init__(self, name: str | None, output_filter: Callable|None = None):
        """
        Initialize the experiment
        """
        self.root: Experiment.Node | None = None
        self.name = name
        self.default_names = defaultdict(int)
        self.all_names: list[str] = []
        self.all_nodes: dict[str, Experiment.Node] = {}
        self.output_filter = output_filter

    def set_output_filter(self, output_filter: Callable|None):
        """
        Sets a filter to apply to output before returning it.
        Individual stages can have their own filters applied to them;
        this filter, if set, is only applied to stages that do not already have
        a filter.
        The filter can be any callable that takes a PipelineData, and
        should return the same type.
        """
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
        """
        Access a stage's node by name.
        """
        return self.all_nodes[node_name]

    def debug_stage(self, node_name, enable=True):
        """
        Turn on debugging for the given stage. The code will enter the
        debugger when the stage's `execute` method is run.
        """
        self.all_nodes[node_name].debug = enable

    def _check_name(self, stage: Stage, name: str | None) -> str:
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
        name = self._check_name(stage, name)
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

    @staticmethod
    def _push_exec(exec_stack: list[Experiment._ExecItem],
                   stage_node: Experiment.Node,
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

    @staticmethod
    def _prepare_exec(stage_node: Experiment.Node,
                      data: PipelineData,
                      count: int,
                      run_index: dict[str, int]):
        if isinstance(stage_node.stage, RepeatingStage):
            run_index = run_index.copy()
            run_index[stage_node.name] = count
        return Experiment._ExecItem(stage_node,
                                    data,
                                    count,
                                    run_index)

    def _handle_stack_item(self,
                       exec_stack: list[Experiment._ExecItem],
                       cur_item: Experiment._ExecItem,
                       next_result: PipelineData,
                       dry_run: bool = False):
        """
        Once a stack item has been executed (yielding `next_result`)
        this method updates the stack with the next stage(s) to run.
        
        """
        # Parse message results; either they are messages to the
        # run (target == "__exec__") or they are stored for downstream
        # use.
        cur_node, cur_input, cur_count, run_idx = cur_item
        exec_msgs = next_result.target_messages("__exec__", remove=True)
        for msg in exec_msgs:
            if msg.target == "__exec__":
                if msg.name == "repeat":
                    if msg.value:
                        # For repeating stages, replace the node and its input
                        # on the stack
                        Experiment._push_exec(exec_stack,
                                              cur_node,
                                              cur_input,
                                              cur_count+1,
                                              run_idx)
        # Check for messages with unknown targets
        for msg in next_result.messages:
            if msg.target not in self.all_nodes and msg.target != "*":
                raise ValueError(f"Message with unknown target "
                                 f"'{msg.target}' generated by stage "
                                 f"'{cur_node.name}'.")
        for next_node in cur_node.consumers:
            Experiment._push_exec(exec_stack,
                                  next_node,
                                  next_result,
                                  0,
                                  run_idx)
    def _prepare_output(self,
                        exec_item: Experiment._ExecItem,
                        result: PipelineData):
        """
        Given the current stack item and its result, prepare the data to
        be returned as an ExperimentResult.
        """
        cur_node, cur_input, cur_count, run_idx = exec_item
        out_data = result.data
        if cur_node.output_filter is not None:
            out_data = cur_node.output_filter(out_data)  
        elif self.output_filter:
            out_data = self.output_filter(out_data)
        out_result = ExperimentResult(out_data,
                                      cur_node.output_name,
                                      run_idx,
                                      cur_input.messages[:],
                                      result.states
                                      )
        return out_result


    def _check_repeat(self, exec_msgs):
        for msg in exec_msgs:
            if msg.target == "__exec__":
                if msg.name == "repeat":
                    if msg.value:
                        return True
        return False

    
    @staticmethod
    def _mp_worker(exp, exec_q, out_q, map_func, logger):
        import multiprocessing as mp
        logger.info(f"Spawned: {mp.current_process()}")
        for fork_limit, exec_item in iter(exec_q.get, CEASE_VALUE):
            logger.info(f"Received: {mp.current_process()}. "
                        f"Fork limit: {fork_limit}, Item: {exec_item[3]}")
            
            try:
                exec_stack : list[Experiment._ExecItem] = [exec_item]

                while exec_stack:
                    # Run the next stage
                    cur_exec_item = exec_stack.pop()
                    cur_node, cur_input, cur_count, run_idx = cur_exec_item
                    if  fork_limit > 0 and isinstance(cur_node.stage, RepeatingStage):
                        logger.info(f"Forking at node: {cur_node.name}")
                        # We only offload tasks onto other processes when we hit
                        # a repeating stage. All iterates of the stage will
                        # be split to different processes.
                        # To do this, we actually run the repeating stage to
                        # exhaustion. Typically, repeating stages just pass
                        # through the inputs, possibly adding messages for
                        # downstream.
                        #
                        # For each iterate, we gather the output results,
                        # and then enqueue all child nodes.
                        assert cur_count == 0
                        while 1:                            
                            next_result = cur_node._run(cur_input, cur_count)
                            run_idx = run_idx.copy()
                            run_idx[cur_node.name] = cur_count
                            exp_msgs = next_result.target_messages("__exec__", remove=True)
                            if not exp._check_repeat(exp_msgs):
                                # We will handle the last iteration in this
                                # process. Since we have removed the exec
                                # messages, it won't repeat.
                                fork_limit -= 1
                                cur_exec_item = cur_node, cur_input, cur_count, run_idx
                                break

                            for next_node in cur_node.consumers:
                                # If we're not careful, we'll pickle the current
                                # state of the RNG, meaning we'll
                                # get identical results for noise
                                fork_result = next_result.copy(
                                    rng = next_result.rng.spawn(1)[0])
                                queue_item = Experiment._prepare_exec(
                                    next_node, fork_result, 0, run_idx)
                            exec_q.put((fork_limit - 1, queue_item))
                            cur_count += 1                            
                    else:
                        next_result = cur_node._run(cur_input,
                                                    cur_count)
                    
                    exp._handle_stack_item(exec_stack, cur_exec_item, next_result)

                    if cur_node.output_result or len(cur_node.consumers) == 0:
                        out_result = exp._prepare_output(cur_exec_item,
                                                         next_result)
                        if map_func is not None:
                            out_q.put((out_result.name,
                                       tuple(out_result.run_index.items()),
                                       map_func(out_result)))
                        else:
                            out_q.put((out_result.name,
                                       tuple(out_result.run_index.items()),
                                       out_result))


            except Exception as e:
                logger.error(f"Encountered an exception: {e}")
                raise
            finally:
                exec_q.task_done()
            # Following is only really needed for testing in single-proc mode
            
            if exec_q.empty():
                return
        
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

        if messages is None:
            messages = []
        elif not isinstance(messages, list):
            messages = [messages]
        for msg in messages:
            if msg.target not in self.all_nodes and msg.target != "*":
                raise ValueError(f"Message with unknown target "
                                 f"'{msg.target}'.")

        init_result = PipelineData(StageData(), messages=messages, rng=rng)

        # Prime the traversal with the root node and initial result.
        Experiment._push_exec(exec_stack, self.root, init_result, 0, {})

        while exec_stack:
            step_count += 1
            # Run the next stage
            cur_exec_item = exec_stack.pop()
            cur_node, cur_input, cur_count, run_idx = cur_exec_item
            if not dry_run or isinstance(cur_node.stage, RepeatingStage):
                next_result = cur_node._run(cur_input,
                                            cur_count)
            else:
                next_result = cur_input

            self._handle_stack_item(exec_stack, cur_exec_item, next_result)

            if cur_node.output_result or len(cur_node.consumers) == 0:
                if dry_run:
                    yield run_idx
                else:
                    yield self._prepare_output(cur_exec_item, next_result)

    def map_mp(self,
               messages=None,
               rng_or_seed: np.random.Generator | int | None = None,
               map_func=None,
               mp_mode="fork",
               num_procs=None,
               fork_limit=2):
        """Run the experiment.
        
        """
        from multiprocessing import get_context, get_logger

        logger = get_logger()

        rng = RNG(rng_or_seed)
        # Essentially, this is just a depth-first traversal of the
        # experiment tree.
        if self.root is None:
            raise ValueError("No stages have been added to experiment.")

        # This is only used to freeze the random number generator in a
        # repeat stage (so that all repetitions have the same random
        # sequence). This functionality probably wont work in mp-mode.
        self._prepare()
        
        if messages is None:
            messages = []
        elif not isinstance(messages, list):
            messages = [messages]
        for msg in messages:
            if msg.target not in self.all_nodes and msg.target != "*":
                raise ValueError(f"Message with unknown target "
                                 f"'{msg.target}'.")

        init_result = PipelineData(StageData(), messages=messages, rng = rng)

        ctx = get_context(mp_mode)
        exec_q = ctx.JoinableQueue()
        result_q = ctx.Queue()
        # Prime the traversal with the root node and initial result. Initially,
        # only one worker will see this
        exec_q.put((fork_limit, (self.root, init_result, 0, {})))

        result = []
        # Start our worker pool
        with ctx.Pool(processes=num_procs,
                      initializer=Experiment._mp_worker,
                      initargs=(self, exec_q, result_q, map_func, logger)) as pool:
            # The pool starts and will processes the pipeline.
            
            # wait for all processes to complete all tasks
            exec_q.join()
            for _ in range(pool._processes):
                exec_q.put(CEASE_VALUE)

            while not result_q.empty():
                result.append(result_q.get(True, .1))
        # Results are out of order, so we sort by run index
        result.sort(key=lambda x: x[1])

        return result

    def map_sp(self,
               messages=None,
               rng_or_seed : np.random.Generator | int | None = None,
               map_func=None,
               fork_limit=2):
        """Run the experiment.
        
        """
        from multiprocessing import JoinableQueue, Queue, get_logger
        logger = get_logger()

        rng = RNG(rng_or_seed)
        # Essentially, this is just a depth-first traversal of the
        # experiment tree.
        if self.root is None:
            raise ValueError("No stages have been added to experiment.")

        # This is only used to freeze the random number generator in a
        # repeat stage (so that all repetitions have the same random
        # sequence). This functionality probably wont work in mp-mode.
        self._prepare()
        
        if messages is None:
            messages = []
        elif not isinstance(messages, list):
            messages = [messages]
        for msg in messages:
            if msg.target not in self.all_nodes and msg.target != "*":
                raise ValueError(f"Message with unknown target "
                                 f"'{msg.target}'.")

        init_result = PipelineData(StageData(), messages=messages, rng=rng)

        exec_q = JoinableQueue()
        result_q = Queue()
        # Prime the traversal with the root node and initial result. Initially,
        # only one worker will see this
        exec_q.put((fork_limit, (self.root, init_result, 0, {})))

        Experiment._mp_worker(self, exec_q, result_q, map_func, logger)

        result = []
        while 1:
            try:
                result.append(result_q.get(True, 0.5))
            except Empty:
                break
        
        return result

    def count_iters(self, messages=None):
        return len(list(self.run(messages=messages,
                                 rng_or_seed=None,
                                 dry_run=True)))

    def run_all(self,
                messages=None,
                rng_or_seed: np.random.Generator | int | None = None,
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

    def run_tree(self,
                 messages=None,
                 rng_or_seed: np.random.Generator | int | None = None,
                 ):
        """Run the experient,returning all the dataframes as output.
        If there is a single output stage, the results are returned as a list.
        Otherwise, the results are grouped by name.
        """
        from tqdm import tqdm
        
        num_out = len(list(self.run(messages=messages,
                                    rng_or_seed=None,
                                    dry_run=True)))

        result = self._build_out_tree()
        with tqdm(total=num_out) as progress_bar:
            for output in self.run(messages, rng_or_seed):
                result.add_output(output)
                progress_bar.update(1)

        return result

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
    def __init__(self, freeze_rng: bool = False, **kwargs):
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
                
        # Pass input through unchanged. All we modify are the messages
        # (and possibly the RNG)
        result = PipelineData(input, msgs, rng)

        return result

    
def quick_run(*stages, min_abundance=0, min_species_presence=0):
    exp = Experiment("temp")
    if any(isinstance(s, RepeatingStage) for s in stages):
        raise ValueError("Quick runs can not include repeating stages")
    exp.set_stages(*stages)
    exp.set_output_filter(OutputFilter(
        min_abundance=min_abundance,
        min_species_presence=min_species_presence))
    result = exp.run_all()
    return result[0]


class ImportStage(Stage):
    """ Import data from existing matrix
    """
    PROVIDES = {"abundance", "base_response", "sample_coords", "sample_info", "species_info"}

    def __init__(self,
                 abundance,
                 sample_coords=None,
                 base_abundance=None,
                 spcs_info=None,
                 sample_info=None,
                 transpose=False,
                 rarity_scale=0.1,
                 **kwargs):
        def unify(m) -> np.ndarray:
            if transpose:
                return np.array(m).T
            else:
                return np.array(m)
        super().__init__(**kwargs)
        self.abundance = unify(abundance)
        if base_abundance is not None:
            if base_abundance.shape != abundance.shape:
                raise ValueError("Base abundance matrix must have same shape as abundance.")
            self.base_abundance = unify(base_abundance)
        else:
            # For common species, we can take the probability (for applying presence/absence
            # noise) as just the abundance over max abundance. However, for rare
            # singleton species, this would imply that they are 100% certain.
            # Instead, we scale probability by total count for a species, so that
            # singletons have probability rarity_scale, with maximum probability
            # increasing as 1-exp(-k*count).
            if rarity_scale <=0 or rarity_scale >1:
                raise ValueError("Prob scale must be in (0,1].")
            k = -np.log(1-rarity_scale)
            self.base_abundance = self.abundance / self.abundance.max(axis=0, initial=1)
            self.base_abundance *= 1 - np.exp(-k * (self.abundance>0).sum(axis=0))
        if spcs_info is not None:
            if len(spcs_info) != self.abundance.shape[1]:
                raise ValueError("Species info length does not equal abundance "
                                 f"({len(spcs_info)} vs. {self.abundance.shape[1]}).")
            self.spcs_info = spcs_info.copy()
            self.spcs_info.index.name="species_id"
        else:
            self.spcs_info = pd.DataFrame(index=range(0, self.abundance.shape[1]))
        if sample_info is not None:
            if len(sample_info) != self.abundance.shape[0]:
                raise ValueError("Sample info length does not equal abundance "
                                 f"({len(sample_info)} vs. {self.abundance.shape[0]}).")
            self.sample_info = sample_info.copy()
            self.sample_info.index.name="sample_id"
        else:
            self.sample_info = pd.DataFrame(index=range(0, self.abundance.shape[0]))
        if sample_coords is not None:
            if len(sample_coords) != self.abundance.shape[0]:
                raise ValueError("Sample coords length does not equal abundance "
                                 f"({len(sample_coords)} vs. {self.abundance.shape[0]}).")
            self.sample_coords = sample_coords.copy()
            self.sample_coords.index.name="sample_id"
        else:
            self.sample_coords = pd.DataFrame(
                {"coord": range(self.abundance.shape[0])},
                index=range(self.abundance.shape[0]))
        self.mean_base = self.base_abundance[self.base_abundance>0].mean()


    def default_name(self) -> str:
        return "Import"

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        # do stuff
        result = PipelineData(input.copy(
            abundance=self.abundance,
            base_response=self.base_abundance,
            species_info=self.spcs_info,
            sample_info=self.sample_info,
            sample_coords=self.sample_coords),
                              [],rng)
        

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



__all__ = [
 'ArraySweep',
 'ComboStage',
 'DebugStage',
 'Experiment',
 'ExperimentResult',
 'ImportStage',
 'LinSpace',
 'Message',
 'OutputFilter',
 'ParameterSweep',
 'PipelineData',
 'RandomSweep',
 'RepeatingStage',
 'ReplicateStage',
 'Stage',
 'StageData',
 'StageParameter',
 'SweepVals',
 'dist_parser',
 'quick_run',
]
