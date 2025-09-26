from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Sequence
from abc import ABC, abstractmethod

from .utils import dotdict
from .distributions import (
    gen_param_dist,
    force_range,
    RNG,
    DISTRIBUTIONS,
    Distribution,
    NormalDistribution,
    UniformDistribution,
    LogNormalDistribution,
    LogRandomDistribution
    )
from .gradient_response import BetaResponse
from .mass_distribution import MassDistribution, RandomMasses
from .experiment import Stage, StageParameter, StageData, PipelineData, Message


from ._core import adjust_masses

CIA_DB = "WHOI_CIA_DB_2016_11_21.parquet"


def load_cia_masses():
    data_path = os.path.join(os.path.dirname(__file__), 'data', CIA_DB)
    cia = pd.read_parquet(data_path)
    return cia.index.values

class LandscapeResponse(ABC):
    class ResponseSet(ABC):
        @abstractmethod
        def apply(self, coords: pd.DataFrame) -> np.ndarray:
            pass

    @abstractmethod
    def generate_response(self, index: list,
                          rng: np.random.Generator | None = None
                          ) -> ResponseSet:
        pass


class SpeciesGroupConfig:
    cia_masses = None

    class ResponseSet:
        # SpeciesGroup Response methods:
        def __init__(self,
                     config,
                     A0,
                     responses,
                     masses):
            self.config = config
            self.species = config.species
            self.name = config.name
            self.N = config.N
            self._A0 = A0
            self.responses = responses
            self.masses = masses
            
        def __getitem__(self, gradient):
            if gradient == "A0":
                return self.A0
            return self.responses[gradient]

        @property
        def A0(self):
            return self._A0

        def df(self):
            keys = list(self.responses.keys())
            response = pd.concat([r.df() for r in self.responses.values()],
                                 axis=1, keys=keys)
            response["A0"] = self.A0

            return response

        def _apply_raw(self, coords, responses):
            result = np.ones_like(self._A0)
            for grad_id in responses:
                if grad_id in coords:
                    X = coords[grad_id]
                    response = responses[grad_id]
                    result = result * response.apply(X)
            return result
        
        def base_response(self, coords, beta_mul=1.0, as_df: bool = False):
            in_df = False
            if isinstance(coords, pd.DataFrame):
                coord_index = coords.index
                coords = {grad: coords[grad].values for grad in coords.columns}
                in_df = True

            result = np.ones_like(self._A0)
            for grad_id, response in self.responses.items():
                if grad_id in coords:
                    X = coords[grad_id]
                    result = result * response.apply(X, beta_mul=beta_mul)

            if in_df and as_df:
                result = pd.DataFrame(result,
                                      index=coord_index,
                                      columns=self.species)
                result.index.name = "sample_id"
                result.columns.name = "species_id"
            return result

        
        def get_abundance(self, coords):
            return  self._A0 * self.base_response(coords)


    # SpeciesGroupConfig        
    def __init__(self,
                 name: str,
                 species: int,
                 A0: Distribution,
                 mass_dist: MassDistribution|None=None,
                 ):
        """Create a new group of species for simulation.

        A species group consists of a list of species, as well as a
        GradientResponse configuration (distribution of response
        parameters for each gradient).            

        Parameters
        ----------
        name : str
            Name of group
        species : int, list, or dict
            Species names. If an int, will generate a list of that many
            species, named '0', '1', ....
        A0 : None, Distribution, or list
            Modal abundance curve for the species.
            If a Distribution, then the modal abundances will be generated
            randomly, using that distribution.
            If None, a default distribution (lograndom(10,100) is used
            instead.
            When abundance modes are specified directly (with a list),
            the size must match the number of species.

        """
        self.name = name
        if isinstance(species, int):
            self.N = species
            self.species = [f"{name}_{n+1}" for n in range(self.N)]
            self.generated_species = True
        else:
            self.species = species
            self.N = len(species)
            self.generated_species = False
        if A0 is None:
            self._A0 = gen_param_dist("A0")
        elif isinstance(A0, Distribution):
            self._A0 = A0
        else:
            if not isinstance(A0, Sequence):
                raise ValueError("'A0' must be a Distribution, a sequence,"
                                 " or None")
            if len(A0) != self.N:
                raise ValueError(f"Length of 'A0' ({len(A0)}) does not "
                                 f"match number of species ({self.N}).")
            self._A0 = np.array(A0)
            
        # Configure mass info
        if mass_dist is None:
            mass_dist = RandomMasses()
        self.mass_dist = mass_dist


        self.gradient_configs = {}

    @property
    def state(self):
        grad_states = dotdict()
        for grad_id in self.gradient_configs:
            grad_states[grad_id] = self.gradient_configs[grad_id].state

        if isinstance(self.A0, Distribution):
            A0_state = self.A0.state
        else:
            A0_state = State("FixedArray", "A0", self.A0)
        if self.generated_species:
            species_state = State("GenSpecies", "N", self.N)
        else:
            species_state = State("FixedArray", "", self.species)
        params = dotdict(
            gradient_responses=grad_states,
            A0=A0_state,
            species=species_state)
        return State("SpeciesGroupResponse", self.name, params)

    def species_info(self):
        result = pd.DataFrame(index=self.species)
        result["group"] = self.name
        result.index.name = "species_id"
        return result

    def check_gradients(self, grads):
        for grad in grads:
            if grad not in self.gradient_configs:
                raise ValueError(f"SpeciesGroup '{self.name}' contains no "
                                 f"sampling strategy for gradient '{grad}'.")

        for grad in self.gradient_configs:
            if grad not in grads:
                raise ValueError(f"SpeciesGroup '{self.name}' contains a"
                                 "sampling strategy for unknown gradient "
                                 f"'{grad}'.")

    @property
    def A0(self):
        return self._A0

    def add_gradient_response(self, grad_id, config):
        self.gradient_configs[grad_id] = config
        return self

    def adjust(self, mode, range, beta, grad_id=None) -> SpeciesGroupConfig:
        gconfs = {}
        for resp_grad, resp in self.gradient_configs.items():
            if grad_id is None or grad_id == resp_grad:
                gconfs[resp_grad] = resp.adjust(mode, range, beta)
            else:
                gconfs[resp_grad] = resp
        result = SpeciesGroupConfig(self.name,
                                    self.N,
                                    self.A0,
                                    self.mass_dist)
        result.gradient_configs = gconfs
        return result

    @staticmethod
    def BasicConfig(name, gradients, N):
        result = SpeciesGroupConfig(name, N, LogRandomDistribution(1e4, 1e9))
        for grad_id in gradients:
            result.add_gradient_response(grad_id, BetaResponse.DefaultResponse())
        return result


    def generate_response(self, rng=None):
        rng = RNG(rng)
        if isinstance(self.A0, Distribution):
            A0 = self.A0(self.N, rng)
        else:
            A0 = self.A0

        responses = {
            grad_id: config.generate_response(self.species, rng=rng)
            for grad_id, config in self.gradient_configs.items()
        }

        masses = self.mass_dist(self.N, rng)
        return SpeciesGroupConfig.ResponseSet(self, A0, responses, masses)

    
class GenSpeciesStage(Stage):
    """
    Stage for generating species responses.

    The constructor takes a list of SpeciesGroupConfigs, which provide
    the number of species and distributions for their fundamental niches,
    Upon execution, this stage randomly selects the niche for each species.
    While the basic configuration must be provided at initialization, it
    can be modified to alter the niche distributions with an
    "adjust_responses" message.
    """

    PROVIDES = ["responses"]
    
    class State(Stage.State):
        def __init__(self, stage: GenSpeciesStage):
            super().__init__(stage)
            self.species_configs = stage.species_configs[:]

        def handle_message(self, message: Message) -> bool:
            if super().handle_message(message):
                return True
            if message.name == "adjust_responses":
                if not isinstance(message.value, dict):
                    raise TypeError("The 'adjust_responses' message must be "
                                    "a dict.")
                value = message.value
                # Check keys for validity. Probably we could centralize
                # message validation, but for now we'll do it ad hoc
                wrong = set(value.keys()).difference([
                    "grad_id",
                    "species_group",
                    "mode",
                    "range",
                    "beta"])
                if len(wrong) > 0:
                    wrong = ", ".join(wrong)
                    raise ValueError("Unexpected key(s) in 'adjust_responses' "
                                     f"message: {wrong}.")
                grad_id = message.value.get("grad_id")
                if grad_id == "*": grad_id = None
                sp_group = message.value.get("species_group")
                if sp_group == "*": sp_group = None
                # Replace any targeted species group config with the
                # adjusted one.
                for idx, cfg in enumerate(self.species_configs):
                    if sp_group is None or cfg.name == sp_group:
                        self.species_configs[idx] = (
                            cfg.adjust(
                                mode=value.get("mode", 0),
                                range=value.get("range", 0),
                                beta=value.get("beta", 0)))
                return True
            # Unhandled message:
            return False
        
    def default_name(self):        
        return "GenSpecies"
                 
    def __init__(self,
                 species_config: SpeciesGroupConfig|list[SpeciesGroupConfig],
                 **kwargs):
        super().__init__(**kwargs)
        if not isinstance(species_config, list):
            self.species_configs = [species_config]
        else:
            self.species_configs = species_config

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: State) -> PipelineData:
        responses = []
        for sgc in state.species_configs:
            responses.append(sgc.generate_response(rng))

        result = PipelineData(input.copy(responses=responses),
                              messages=[],
                              rng=rng)
        return result
            

def apply_core_sim(input: StageData) -> StageData:
    # All this is is taking all the species response sets and sampling them
    # at the given coordinates.
    # We store both the base (0 to 1) response for use in noise stages,
    # as well as the abundance.
    base_responses = []
    for response in input.responses:
        base_responses.append(response.base_response(input.sample_coords))

    base_response = np.concat(base_responses, axis=1)
    abundance = np.concat([resp.A0 * base_response
                           for resp, base_response in zip(input.responses,
                                                          base_responses)],
                          axis=1)
    # Finally, we gather all the species info together in a dataframe.
    species_info = pd.concat([
        pd.DataFrame(dict(group=response.name, mass=response.masses),
                     index=response.species)
        for response in input.responses])
    species_info.index.name = "species_id"
    
    return input.copy(abundance=abundance,
                      base_response=base_response,
                      species_info=species_info)
    

class GenCoreStage(Stage):
    REQUIRES = ["sample_coords", "sample_info", "responses"]
    PROVIDES = ["abundance", "base_response", "species_info"]

    def default_name(self) -> str:
        return "GenerateCore"

    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        return PipelineData(apply_core_sim(input), messages=[], rng=rng)

class MockCoreStage(Stage):
    PROVIDES = ["abundance", "base_response", "species_info", "sample_coords"]

    def default_name(self):
        return "MockCore"

    def __init__(self, abundance,
                 base_response=None,
                 species_info=None,
                 sample_info=None,
                 sample_coords=None):
        super().__init__()
        if base_response is None:
            base_response = abundance > 0
        if sample_info is None:
            sample_info = pd.DataFrame(index=np.arange(abundance.shape[0]))
            sample_info.index.name = "sample_id"
        if species_info is None:
            species_info = pd.DataFrame(index=np.arange(abundance.shape[1]))
            species_info.index.name = "species_id"
        if sample_coords is None:
            sample_coords = pd.DataFrame(np.arange(abundance.shape[0]),
                                         index=sample_info.index)
        self.data = StageData(abundance=abundance,
                              base_response=base_response,
                              species_info=species_info,
                              sample_info=sample_info,
                              sample_coords=sample_coords)
    def execute(self, input: StageData,
                rng: np.random.Generator,
                state: Stage.State) -> PipelineData:
        return PipelineData(self.data.copy(), messages=[], rng=rng)

        

    
