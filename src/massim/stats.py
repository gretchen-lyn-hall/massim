from .experiment import ExperimentResult
from ._core import compute_aff, jaccard, braycurtis
from .utils import PrettyDict

import numpy as np
import pandas as pd

def delete_diagonal(array):
    """Remove all diagonal elements of an NxN matrix and return an
    Nx(N-1) matrix (preseeve size of axis 0)"""
    return array[~np.eye(len(array), dtype=bool)].reshape(len(array), -1)


def affinity_regressions(sim_rows, aff_rows):
    import scipy.stats as ST
    coords = np.array([sim_rows, aff_rows])
    order = np.lexsort(coords)
    coords = coords[:, order]
    regressions = []
    for ii in range(coords.shape[1] - 3):
        try:
            regressions.append(
                ST.linregress(coords[0, ii:],
                              coords[1, ii:]))
        except ValueError:
            # We can sometimes run into issues, especially when collecting
            # permuted stats, where many similarity values are equal.
            # Ignore if we can regress on at least half the points
            if ii < coords.shape[1]/2:
                raise
    regress = pd.DataFrame(regressions)
    best_regress = np.argmin([r.pvalue for r in regressions])
    regress.attrs["discontinuous_sites"] = best_regress
    regress.attrs["discontinuous_indices"] = order[:best_regress]
    return regress

def stats_shannon(exp_result: ExperimentResult):
    mat = exp_result.abundance
    # get proportions:
    mat = mat / mat.sum(axis=1)[:, None]
    presence = mat > 0
    s_vals = mat[presence]
    s_vals = -np.log(s_vals) * s_vals
    mat[presence] = s_vals
    return pd.Series(mat.sum(axis=1), index=exp_result.sample_info.index)

def chou(exp_result: ExperimentResult):
    mat = exp_result.abundance
    mat = mat / mat.sum(axis=1)[:, None]
    presence = mat > 0
    s_vals = mat[presence]
    s_vals = -np.log(s_vals) * s_vals
    mat[presence] = s_vals
    return pd.Series(mat.sum(axis=1), index=exp_result.sample_info.index)
    

def generalized_diversity(exp_result: ExperimentResult, qs: float | list[float]):
    prop = exp_result.abundance
    # get proportions:
    prop = prop / prop.sum(axis=1)[:, None]
    presence = prop > 0
    temp = np.zeros_like(prop)

    # Taken from Patil and Taillie 1982, though here we use q starting from 0
    # instead of 1.
    
    def single_value(q):
        s_vals = prop[presence]
        if q == 0:
            return presence.sum(axis=1)-1
        elif q == 1:
            s_vals = -np.log(s_vals) * s_vals
        else:
            s_vals = s_vals**q
            
        temp[presence] = s_vals
        if q == 1:
            div = temp.sum(axis=1)
        else:
            div = (1 - temp.sum(axis=1)) / (q - 1)
        return div

    if isinstance(qs, (list, tuple, np.ndarray)):
        div = pd.DataFrame({q: single_value(q) for q in qs})
        div.columns = [f"div_q{q}" for q in qs]
        return div
    else:
        div = single_value(qs)
        return pd.Series(div, index=exp_result.sample_info.index)


def hill_numbers(exp_result: ExperimentResult, qs: float|list[float]):
    prop = exp_result.abundance

    # get proportions:
    prop = prop / prop.sum(axis=1)[:, None]
    presence = prop > 0
    temp = np.zeros_like(prop)

    # Taken from Patil and Taillie 1982, though here we use q starting from 0
    # instead of 1.
    def single_value(q):
        s_vals = prop[presence]
        if q == 0:
            s_vals = np.ones_like(s_vals)
        elif q == 1:
            s_vals = s_vals * np.log(s_vals)
        else:
            s_vals = s_vals**q
            
        temp[presence] = s_vals
        if q == 1:
            div = np.exp(-temp.sum(axis=1))
        else:
            div = temp.sum(axis=1) ** (1/(1 - q))
        return div
            
    if isinstance(qs, (list, tuple, np.ndarray)):
        div = pd.DataFrame({q: single_value(q) for q in qs})
        div.columns = [f"div_q{q}" for q in qs]
        return div
    else:
        div = single_value(qs)
        return pd.Series(div, index=exp_result.sample_info.index)

def evenness(exp_result: ExperimentResult, qs: float|list[float]):
    div = hill_numbers(exp_result, qs)
    if isinstance(qs, (list, tuple, np.ndarray)):
        rich = (exp_result.abundance > 0).sum(axis=1)
        even = div.div(rich, axis=0)
        even.columns = [f"even_q{q}" for q in qs]
        return even
    else:
        rich = (exp_result.abundance > 0).sum(axis=1)
        even = div / rich
        return pd.Series(even, index=exp_result.sample_info.index,
                         name=f"even_q{qs}")
    

def affinity_matrix(exp_result: ExperimentResult, dist_method: str = "jaccard", mask=None):
    abd = exp_result.abundance
    idx = exp_result.sample_info.index
    if mask is not None:
        abd = np.delete(abd, mask, axis=0)
        idx = np.delete(idx, mask)
    if dist_method == "jaccard":
        dist = jaccard(abd.T > 0)
    elif dist_method == "braycurtis":
        dist = braycurtis(abd.T)
    else:
        raise ValueError(f"Unknown distance method '{dist_method}'")
    aff = compute_aff(dist)
    aff_rows = delete_diagonal(aff).mean(axis=1)
    sim_rows = delete_diagonal(dist).mean(axis=1)
    return pd.DataFrame(dict(aff=aff_rows, sim=sim_rows),
                            index=idx)


def mosaic_diversity(exp_result: ExperimentResult,
                     dist_method='jaccard',
                     affinity=None):
    if affinity is None:
        affinity = affinity_matrix(exp_result, dist_method=dist_method)
    regressions = affinity_regressions(sim_rows=affinity.sim,
                                       aff_rows=affinity.aff)
    full_regress = regressions.iloc[0]
    best_regress = regressions.iloc[np.argmin(regressions.pvalue)]
    discon_indices = []
    if regressions.attrs["discontinuous_sites"] > 1:
        import scipy.stats as ST
        discon_indices = regressions.attrs["discontinuous_indices"]
        # Rerun affinity excluding discontinuous
        affinity2 = affinity_matrix(exp_result,
                                    dist_method=dist_method,
                                    mask=regressions.attrs["discontinuous_indices"])
        redo_regress = ST.linregress(affinity2.sim, affinity2.aff)
        redo_sim = affinity2.sim.mean()
    else:
        redo_regress = best_regress
        redo_sim = affinity.sim.mean()
    

    # In the regressions dataframe, sites are ordered w.r.t. mean sim,
    # and the slopes are calculated for sites 0-n, 1-n, 2-n, ...
    # Sites which are 'discontinuous' (bend away from the linear region)
    # typically occur at low similarity values.
    # We find the linear region by finding the point with minimum p-value.
    # Sites to the left of that are then non-linear.
        
    best_idx = regressions.attrs["discontinuous_sites"]
    result = PrettyDict()
    result.mu_all = full_regress.slope
    result.mu_best = best_regress.slope
    result.mu_redo = redo_regress.slope
    result.discontinuous_sites = best_idx
    result.discontinuous_indices = discon_indices
    result.best_regress = best_regress
    result.full_regress = full_regress
    result.redo_regress = redo_regress
    result.mean_sim = affinity.sim.mean()
    result.mean_sim_redo = redo_sim
    return result

def mosaic_diversity_basic(exp_result: ExperimentResult,
                     dist_method='jaccard',
                     affinity=None):
    if affinity is None:
        affinity = affinity_matrix(exp_result, dist_method=dist_method)
    import scipy.stats as ST
    reg = ST.linregress(affinity.sim, affinity.aff)

    # In the regressions dataframe, sites are ordered w.r.t. mean sim,
    # and the slopes are calculated for sites 0-n, 1-n, 2-n, ...
    # Sites which are 'discontinuous' (bend away from the linear region)
    # typically occur at low similarity values.
    # We find the linear region by finding the point with minimum p-value.
    # Sites to the left of that are then non-linear.
        
    result = PrettyDict()
    result.mu = reg.slope
    result.mean_sim = affinity.sim.mean()
    return result

class BasicStats:
    def __init__(self, dist_method = "jaccard"):
        self.dist_method = dist_method    


    def __call__(self, exp_result: ExperimentResult):
        result = PrettyDict()
        
        abundance = exp_result.abundance
        presence = abundance != 0
        
        richness = presence.sum(axis=1)
        result.richness_min = richness.min()
        result.richness_max = richness.max()
        result.richness_mean = richness.mean()
        result.add_spacer()
        site_abundance = abundance.sum(axis=1)
        result.set_round(0)
        result.abundance_min = site_abundance.min()
        result.abundance_max = site_abundance.max()
        result.total_species = (presence.sum(axis=0)>0).sum()

        shan = stats_shannon(exp_result)
        result.add_spacer()
        result.set_round(3)
        result.shannon_mean = shan.mean()
        result.shannon_std = shan.std()

        result.add_spacer()
        result.set_round(4)
        bray = braycurtis(abundance.T)
        bray_rows = delete_diagonal(bray).mean(axis=1)

        result.beta_sim_bray = bray_rows.mean()
        result.beta_sim_bray_var = bray_rows.var()

        result.add_spacer()
        jacc = jaccard(abundance.T)
        jacc_rows = delete_diagonal(jacc).mean(axis=1)

        result.beta_sim_jacc = jacc_rows.mean()
        result.beta_sim_jacc_var = jacc_rows.var()

        if self.dist_method == 'jaccard':
            aff = compute_aff(jacc)
        else:
            aff = compute_aff(bray)

        aff_rows = delete_diagonal(aff).mean(axis=1)
        result.add_spacer()

        try:
            regressions = affinity_regressions(sim_rows=jacc_rows,
                                               aff_rows=aff_rows)
            full_regress = regressions.iloc[0]
            best_regress = regressions.iloc[regressions.pvalue.argmin()]

            result.mosaic_overall = best_regress.slope
            result.mosaic_linear = full_regress.slope
            result.affinity_var = aff_rows.var()
        except ValueError as e:
            result.mosaic_error = f"Regression error: {e}"
        return result

