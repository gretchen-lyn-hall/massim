import massim.all as ma
import numpy as np
import pandas as pd
from importlib.resources import files

DATA_FILES = files("massim.data").joinpath("emerge_stats")



def test_run_simple():
    N_spcs = 500
    N_samp = 50
    N_total = 28_000
    csamp, cspcs, core = ma.quick_compas(N_spcs, N_samp, 3, sample_method="random")

    xfrm_non = pd.read_csv(DATA_FILES.joinpath("transform_stats_nr.csv"))
    metabs = pd.read_csv(DATA_FILES.joinpath("masses.csv"))

    # For resampling intensities, we 
    intens_icdf2D = pd.read_csv(DATA_FILES.joinpath("intensity_distributions.csv"),
                                index_col=0).values

    intens_dist2D = ma.PowDistribution(10,
                                       ma.OneOfDistribution(
                                           [ma.PiecewiseEmpiricalDistribution(row)
                                            for row in intens_icdf2D]))



    pconf = ma.ProfileGroupConfig(N_spcs,
                                  N_total,
                                  xfrm_non,
                                  initial_components=1,
                                  initial_masses=ma.ChoiceDistribution(metabs.mass.values[:500:5]),
                                  initial_intensities=1.0,
                                  intensity_scale=ma.UniformDistribution(0,1),
                                  min_intensity=1e-6,
                                  tolerance_ppm=1.0,
                                  weight_exponent=4,
                                  pre_weight=1)

    p_gen_stage = ma.GenProfileStage(pconf)
    p_apply_stage = ma.ApplyProfileStage()

    ithresh = ma.AdaptiveIntensityThreshold(target_presence=2500/N_total)

    resamp_stage = ma.ResampleIntensityStage(intens_dist2D)
    ep = ma.quick_run(p_gen_stage,csamp, cspcs, core, p_apply_stage, ithresh,
                      resamp_stage, min_species_presence=1)
    assert ep.abundance.shape[0] == 50
    assert ep.abundance.shape[1] > 1000
