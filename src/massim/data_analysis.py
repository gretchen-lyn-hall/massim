import pandas as pd
import numpy as np
import igraph as ig
import tqdm

from .utils import dotdict

class DataAnalysis:
    def __init__(self):
        self.graphs = []
        self.xfrms = []
        self.counts = None
        self.keys = None
        self.presence = None


def detect_axis(mat: pd.DataFrame|np.ndarray,
                compound_info: pd.DataFrame|pd.Series|np.ndarray,
                compound_axis: int|None) -> int:
    if compound_axis is None:
        if compound_info.shape[0] == mat.shape[0]:
            return 0
        elif compound_info.shape[0] == mat.shape[1]:
            return 1
        else:
            raise ValueError("Intensity matrix not aligned with mass list")
    elif compound_info.shape[0] != mat.shape[compound_axis]:
            raise ValueError("Intensity matrix not aligned with mass list")
    return compound_axis

def build_sample_graph(masses: pd.Series|np.ndarray,
                       compound_ids: np.ndarray,
                       keys: pd.DataFrame,
                       key_column: str = "mass_delta",
                       err_abs = 0.001,
                       **vertex_data
                       ):
    """Find  potential transforms in a single sample."""
    from ._core import find_transforms
    
    xfrms = np.array(find_transforms(masses, keys[key_column], err_abs))
    if len(xfrms) > 0:
        xfrm_ids = xfrms[:, 0]
        edges = xfrms[:, 1:]
    else:
        xfrm_ids = np.array([], dtype=np.int32)
        edges = []
    
    G = ig.Graph(masses.shape[0],
                 edges)
    for attr_name in vertex_data:
        G.vs[attr_name] = vertex_data[attr_name]
    G.vs["name"] = compound_ids
    G.vs["compound_id"] = compound_ids
        
    edgemap = keys.iloc[xfrm_ids]
    for col in keys.columns:
        G.es[col] = edgemap[col].values
    G.es["name"] = xfrm_ids
    G.es["xfrm_id"] = xfrm_ids

    return xfrms, G

def get_intensity_distributions(intensity: pd.DataFrame|np.ndarray):
    pass

    

def get_dataset_transforms(intensity: pd.DataFrame|np.ndarray,
                           compound_info: pd.DataFrame|pd.Series|np.ndarray,
                           keys: pd.DataFrame,
                           err_abs=0.001,
                           mass_column="mass",
                           key_column='mf',
                           compound_axis=None):
    """Return a dict of sample_id -> raw transform"""
    from ._core import find_transforms
    # For speed and convenience, convert to numpy
    intensity = np.array(intensity)

    # If compound info is a single array/series, assume it is mass
    if isinstance(compound_info, (np.ndarray|pd.Series)):
        compound_dict = {mass_column: np.array(compound_info)}
        compound_id = np.arange(len(compound_info))
    else:
        compound_dict = {col: compound_info[col].values for col in compound_info}
        compound_id = compound_info.index

    if isinstance(keys, (np.ndarray|pd.Series)):
        keys = pd.DataFrame(keys, columns=[key_column])
    
    masses = compound_dict[mass_column]

    compound_axis = detect_axis(intensity, compound_info, compound_axis)

    result = DataAnalysis()
    result.keys = keys.rename(columns={key_column: "mass_delta"})
    result.counts = np.zeros((len(keys), intensity.shape[1-compound_axis]),
                             np.int32)
    result.presence = intensity > 0
    
    for samp_id in tqdm.trange(intensity.shape[1-compound_axis],
                               desc="Computing transformations"):
        if compound_axis == 0:
            sample = intensity[:, samp_id]
        else:
            sample = intensity[samp_id, ]

        presence = sample > 0            
        samp_mass = masses[presence]
        samp_intens = sample[presence]
        vertex_data = {col: compound_dict[col][presence]
                       for col in compound_dict}

        xfrms, G = build_sample_graph(samp_mass,
                                      compound_id[presence],
                                      keys,
                                      key_column=key_column,
                                      err_abs=err_abs,
                                      intensity=samp_intens,
                                      **vertex_data)
                                      
        ct_id, ct_num = np.unique(xfrms[:,0], return_counts=True)
        result.counts[ct_id, samp_id] = ct_num
        result.xfrms.append(xfrms)
        result.graphs.append(G)
        
    return result


def transform_stats(data: DataAnalysis):
    assert data.keys is not None

    # First, compute the mean lengths of transformation chain
    mean_lens = np.zeros((data.keys.shape[0], len(data.graphs)))
    chain_counts = np.zeros((data.keys.shape[0], len(data.graphs)))

    for xfrm_id in tqdm.trange(data.keys.shape[0], desc="Commputing chains"):
        for samp_id, G in enumerate(data.graphs):
            esub = G.subgraph_edges(G.es.select(xfrm_id=xfrm_id))
            # Get the mean chain length for each transformation
            chain_lens = esub.connected_components().sizes()
            mean_lens[xfrm_id, samp_id] = np.mean(chain_lens)
            chain_counts[xfrm_id, samp_id] = len(chain_lens)
    mean_lens = pd.DataFrame(mean_lens, index=data.keys.index)

    # Next, we want to know how many chains there are per metabolite
    chain_counts = pd.DataFrame(chain_counts / data.presence.sum(axis=0),
                                index=data.keys.index)

    result = data.keys.copy()
    result["chain_len"] = mean_lens.mean(axis=1)
    result["chain_len_sd"] = mean_lens.std(axis=1)
    result["chain_prob"] = chain_counts.mean(axis=1)
    result["chain_prob_sd"] = chain_counts.std(axis=1)
    return result


def _find_nonredundant_step(xkeys,
                            mass_col='mass_delta',
                            min_thresh=1,
                            err_abs=2e-5,
                           ):
    
    from ._core import find_transforms
    
    # Ensure transforms are sorted by increasing mass
    xkeys = xkeys.sort_values(mass_col).reset_index(drop=True)

    # Build a transformation network where the nodes and edges come from the
    # xfrm list.

    xauto, G = build_sample_graph(xkeys[mass_col],
                                  xkeys._orig_id,
                                  xkeys,
                                  key_column=mass_col,
                                  err_abs=err_abs)

    weights = xkeys["_weight"][G.es['xfrm_id']]

    # Find a minimal spanning forest and get the edge ids from it
    # By weighting by mass, we get a small (though perhaps not ideally small)
    # set of transforms
    min_t = G.spanning_tree(weights=weights)
    min_edges = set(min_t.es['xfrm_id'])

    # Now, for each connected component, if *none* of the nodes are in our
    # transform set, we include one node (the one with the lowest
    # weight). Since the component is spanned by the min_tree, this is
    # enough to insure all transforms are reachable.
    comp_xfrms = []
    for cc in G.connected_components():
        if not min_edges.intersection(cc):
            node_weights = [(xkeys._weight.loc[node], node) for node in cc]
            comp_xfrms.append(min(node_weights)[1])
    assert len(min_edges.intersection(comp_xfrms)) == 0

    
    min_set = sorted(list(min_edges.union(comp_xfrms)))
    result = xkeys.loc[min_set]
    # Return the minimal subset of our transformation, in sorted order,
    # threshholded 
    return result[result._count>=min_thresh].reset_index(drop=True)

def find_nonredundant_xfrms(xkeys,
                            mass_col='mf',
                            counts=None,
                            weight_by="mass",
                            invert_weights=False,
                            min_thresh=0,
                            err_abs=2e-5):
    """
    Given a set of transformations, find a nonredundant subset, such that no transformation
    in the set is a simple combination of other transformations.

    To determine which transforms to keep, transformations can be given weights such that
    transformations with lower weights are more likely to be preserved.
    The values for 'weight_by' can be:
      'mass' (default): low mass transformations are preferred.
      'counts': A count matrix with the same number of rows as the transformations must be
                supplied. High-count transformations are preferred
      <string>: Use the name of a column in xkeys; by default, the lower values are
                preferred. Set 'invert_weights' to True to change this.
      None: No weighting is performed.

    """
    xkeys = xkeys.copy()
    
    if counts is not None:
        if counts.shape[0] != len(xkeys):
            raise ValueError("Mismatch between transformations and counts.")
        csum = counts.sum(axis=1)
        csum = csum / csum.sum()
        xkeys["_count"] = csum
    else:
        xkeys["_count"] = min_thresh

    if weight_by == "mass":
        xkeys['_weight'] = xkeys[mass_col]
    elif weight_by == "count":
        if counts is None:
            raise ValueError("Must supply count matrix when weight_by=='counts'.")
        xkeys['_weight'] = 1 - xkeys._count
    elif weight_by is not None:
        xkeys['_weight'] = xkeys[weight_by]
        if invert_weights:
            xkeys['_weight'] = xkeys['_weight'].max() - xkeys['_weight']
    else:
        xkeys['_weight'] = 1
        
        
    
    last_size = len(xkeys) + 1
    cur_xfrms = xkeys.copy()
    print("Iteration 0   Size=", len(cur_xfrms))
    idx = 0
    cur_stable = False
    while not cur_stable:
        last_size = len(cur_xfrms)
        cur_xfrms = _find_nonredundant_step(cur_xfrms,
                                            mass_col=mass_col,
                                            min_thresh=min_thresh,
                                            err_abs=err_abs)
        idx += 1
        print("Iteration ", idx, "   Size=", len(cur_xfrms))
        if len(cur_xfrms) == last_size:
            cur_stable = True
    return cur_xfrms.drop(['_weight', '_count'], axis=1)
