from enum import Flag
import pandas as pd
import numpy as np

from .sampling import SamplerConfig
from .species_config import SpeciesGroupConfig

class SimMode(Flag):
    GEN_RESPONSE = 0b000001
    GEN_COORDS = 0b000010
    QUANT_NOISE = 0b000100
    QUAL_NOISE = 0b001000
    TRANSFORMS = 0b010000

    BASE = 0b000011
    ALL = 0b011111

class Simulator:
    class ResultStage:
        def __init__(self, result, data, rng):
            self.data = data
            self.rng_state = rng.bit_generator.state

        def restore_state(self, rng):
            rng.bit_generator.state = self.rng_state
    
    class Result:
        def __init__(self, sim: Simulator,
                     sim_mode: SimMode,
                     responses: list[SpeciesGroupConfig.ResponseSet] | None = None,
                     coords: pd.DataFrame | None = None,
                     rng: np.random.Generator | None = None):
            self.sim = sim
            self.response_curves = responses
            if self.response_curves is None:
                self.response_curves = sim._gen_response_curves(rng=rng)
            self.coords = coords
            if self.coords is None:
                self.coords = sim._gen_coords(rng=rng)
            
                self.base_results = [rc.apply(self.coords)
                                     for rc in self.response_curves],

            if SimMode.QUANT_NOISE in sim_mode:
                self.quant_results = [
                    sg.apply_presence_absence_noise(dat, self.coords, 
                                                    rng=rng)
                    for dat, sg in zip(self.base_results,
                                       self.sim.species_groups)]
            else:
                self.quant_results = self.base_results

            if SimMode.QUAL_NOISE in sim_mode:
                self.qual_results = [
                    sg.apply_quantitative_noise(dat, self.coords, 
                                                rng=rng)
                    for dat, sg in zip(self.base_results,
                                       self.sim.species_groups)]
            else:
                self.qual_results = self.quant_results

            if SimMode.TRANSFORMS in sim_mode:
                self.sim.apply_transformations()
                


        def recompute(self,
                      sim_mode: SimMode,
                      rng: np.random.Generator | None = None
                      ) -> Simulator.Result:
            if SimMode.GEN_COORDS in sim_mode:
                self.coords = 

    def __init__(self, gradients: list[str],
                 cull_species: bool = True):
        self.gradients = gradients
        self.samplers = []
        self.species_groups = []
        self.transform_groups = []

    def add_sampler(self, sampler: SamplerConfig):
        unknown = sampler.sampled_gradients.difference(self.gradients)
        if unknown:
            unknown = ', '.join(unknown)
            raise ValueError(f"Sampler contains unknown gradients '{unknown}'")
        missing = set(self.gradients).difference(sampler.sampled_gradients)
        if missing:
            missing = ', '.join(missing)
            raise ValueError(f"Sampler missing gradients '{missing}'")
        self.samplers.append(sampler)

    def add_species_group(self, sg: SpeciesGroupConfig):
        sg.check_gradients(self.gradients)
        self.species_groups.append(sg)


    def species_info(self) -> pd.DataFrame:
        result = pd.concat(sg.species_info()
                           for sg in self.species_groups)
        result.index.name = "species_id"
        return result

    def sample_info(self) -> pd.DataFrame:
        return pd.concat(samp.info() for samp in self.samplers)


    def _gen_coords(self,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
        return pd.concat(samp.sample(rng) for samp in self.samplers)

    def _gen_response_curves(self,
                            rng: np.random.Generator | None = None
                            ) -> list[SpeciesGroupConfig.ResponseSet]:
        curves = [resp.generate_response(rng=rng)
                  for resp in self.species_groups]

    def compute(self,
                sim_mode: SimMode,
                rng: np.random.Generator | None = None) -> Simulator.Result:
        if sim_mode.GEN_COORDS not in sim_mode:
            raise ValueError("Initial computation must include coordinate "
                             "generation") 
        if  SimMode.GEN_RESPONSE not in sim_mode:
            raise ValueError("Initial computation must include species"
                             "response curve generation")
        coords = self.gen_coords(rng)
        response_curves = [sg.generate_response(rng=rng)
                           for sg in self.species_groups]

        base_results = [resp.apply(coords) for resp in response_curves]

