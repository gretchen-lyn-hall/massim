import pandas as pd
import numpy as np
import igraph as ig
import tqdm
from copy import copy
from collections import defaultdict
import scipy.stats as ST

from .utils import dotdict
from .graph_util import make_graph, TransformGraph
from ._core import find_transforms, track_transforms, MODE_STRICT, MODE_LAX, MODE_MODERATE

CORE_ERR_MODES = {
    'lax': MODE_LAX,
    'strict': MODE_STRICT,
    'moderate': MODE_MODERATE,
    }

USE_MP = True
MP_MODE = 'fork'
import multiprocessing as mp


DEFAULT_XFRMS = pd.read_csv("~/Tfaily/emerge/mire/mire/transf_key.csv")

class DataAnalysis:
    def __init__(self):
        self.graphs = []
        self.xfrms = []
        self.counts = None
        self.keys = None
        self.presence = None
        self.masses = None
        self.intensity = None

    def spectrum(self, idx):
        intens = self.intensity[:, idx]
        mass = self.masses[intens > 0]
        intens = intens[intens > 0]
        return intens, mass


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

def find_xfrms_py(masses, xkeys, err_abs, allow_overlap=True):
    X, Y = np.triu_indices(len(masses), k=1)
    diffs = masses[Y] - masses[X]
    O = np.argsort(diffs, kind='mergesort')
    diffs = diffs[O]
    X = X[O]
    Y = Y[O]
    c_min = 0
    c_max = len(diffs)
    results = []
    errs = []
    for idx, delta in enumerate(xkeys):
        lo = np.searchsorted(diffs[c_min:], delta - err_abs) + c_min
        hi = np.searchsorted(diffs[c_min:], delta + err_abs) + c_min
        if lo == hi:
            continue
        if lo == c_max:
            break

        results.append(([idx]*(hi-lo),
                        X[lo:hi],
                        Y[lo:hi]))
                       
        errs.append(diffs[lo:hi] - delta)

        if allow_overlap:
            c_min = lo
        else:
            c_min = hi
    if not results :
        return np.array([[],[],[]]).T, np.array([])
    return np.concat(results, axis=1).T, np.abs(np.concat(errs))



class GraphBuilder:
    def __init__(self,
                 keys: pd.DataFrame,
                 key_column: str = "mass_delta",
                 err_abs=0.001,
                 err_ppm=1.0,
                 err_mode="moderate",
                 clean_peaks=False,
                 allow_overlap=False) :
        
        self.key_mass = np.array(keys[key_column])
        self.keys = keys
        if err_mode not in ["strict", "lax", "moderate"]:
            raise ValueError('`err_mode` must be one of: "strict", "lax", "moderate"')
        self.err_abs = err_abs
        self.err_ppm = err_ppm
        self.err_mode = err_mode
        self.clean_peaks = clean_peaks
        self.allow_overlap = allow_overlap


    def find_xfrms_raw(self, masses, err_abs):
        """
        A python version of the transformation finder, for use when
        multiprocessing is in use. Only finds transforms within absolute
        error.
        Returns an array of (xfrm_id, src, dst), and an array of absolute
        errors.
        """
        X, Y = np.triu_indices(len(masses), k=1)
        diffs = masses[Y] - masses[X]
        O = np.argsort(diffs, kind='mergesort')
        diffs = diffs[O]
        X = X[O]
        Y = Y[O]
        c_min = 0
        c_max = len(diffs)
        results = []
        errs = []
        for idx, delta in enumerate(self.key_mass):
            lo = np.searchsorted(diffs[c_min:], delta - err_abs) + c_min
            hi = np.searchsorted(diffs[c_min:], delta + err_abs) + c_min
            if lo == hi:
                continue
            if lo == c_max:
                break
            results.append(([idx]*(hi-lo),
                            X[lo:hi],
                            Y[lo:hi]))

            errs.append(diffs[lo:hi] - delta)

            if self.allow_overlap:
                c_min = lo
            else:
                c_min = hi
        return np.concat(results, axis=1).T, np.abs(np.concat(errs))


    def find_transforms(self,
                        masses: pd.Series|np.ndarray):
        """
        Find transformations using the parameters passed to the constructor.
        Returns a dataframe with columns for xfrm_id, src, dst, and error
        """
        if not isinstance(masses, np.ndarray):
            masses = np.array(masses)

        mass_err = None
        if self.err_ppm is not None:
            # We do this in two stages:
            #  First, we find all transforms using a fixed error set to
            #  `err_ppm` ppm of the largest mass.
            #  Then, find the error of the transforms and filter
            mass_err = masses * self.err_ppm / 1e6
            raw_err = mass_err.max()
        else:
            raw_err = self.err_abs
        """
        # UNUSED: The tracker method is much much faster. Works with MP if
        # necessary. Returns identical results.
        
        if USE_MP:
            xfrms, err = self.find_xfrms_raw(masses, raw_err)
        else:
            xfrms = np.array(find_transforms(masses, self.key_mass,
                                             raw_err, self.allow_overlap))
            if len(xfrms) > 0:
                err = abs((masses[xfrms[:, 2]] - masses[xfrms[:, 1]])
                          - self.key_mass[xfrms[:, 0]])
            else:
                err = np.array()

        if self.err_ppm is not None:
            assert isinstance(mass_err, np.ndarray)
            if self.err_mode == "strict":
                comp_err = mass_err[xfrms[:, 1]]
            elif self.err_mode == "lax":
                comp_err = mass_err[xfrms[:, 2]]
            else:
                comp_err = np.sqrt(mass_err[xfrms[:, 1]] * mass_err[xfrms[:, 2]])

            ok = abs(err) < comp_err
            xfrms = xfrms[ok]
            err = err[ok]
        """
        xfrms = track_transforms(masses, self.key_mass,
                                 self.err_ppm, CORE_ERR_MODES[self.err_mode])
        if len(xfrms) > 0:
            err = abs((masses[xfrms[:, 2]] - masses[xfrms[:, 1]])
                      - self.key_mass[xfrms[:, 0]])
        else:
            err = np.array()



        
        
        xfrms = pd.DataFrame(xfrms, columns=["xfrm_id", "src", "dst"])
        xfrms["err"] = err
        if self.clean_peaks:
            xfrms = slowclean(xfrms)
        return xfrms

    def build_graph_raw(self, xfrms, compound_ids, vertex_data):
        xfrm_ids = xfrms.values[:, 0]
        edges = xfrms.values[:, 1:3]
        edge_data = {"xfrm_id": xfrm_ids}

        G = make_graph(len(compound_ids), edges, vdata=vertex_data, edata=edge_data, use_mp=USE_MP)
        G.add_vertex_attrs(compound_id=compound_ids)
        return G

    def build_graph(self, masses, compound_ids, vertex_data=None):
        if vertex_data is None:
            vertex_data = {}
        # Find potential transforms
        xfrms = self.find_transforms(masses)
        G = self.build_graph_raw(xfrms, compound_ids, vertex_data)
        return xfrms, G




def build_sample_graph(masses: pd.Series|np.ndarray,
                       compound_ids: np.ndarray,
                       keys: pd.DataFrame,
                       key_column: str = "mass_delta",
                       err_abs=0.001,
                       err_ppm=1.0,
                       err_mode="moderate",
                       clean_peaks=False,
                       allow_overlap=False,
                       vertex_data = None,
                       graph_mode="ig"
                       ):
    """Find  potential transforms in a single sample."""
    if vertex_data is None:
        vertex_data = {}

    if not isinstance(masses, np.ndarray):
        masses = np.array(masses)

    if key_column is None:
        assert isinstance(keys, np.ndarray)
        key_mass = keys
    else:
        key_mass = np.array(keys[key_column])
    if err_mode not in ["strict", "lax", "moderate"]:
        raise ValueError('`err_mode` must be one of: "strict", "lax", "moderate"')

    mass_err = None
    if err_ppm is not None:
        # We do this in two stages:
        #  First, we find all transforms using a fixed error set to
        #  `err_ppm` ppm of the largest mass.
        #  Then, find the error of the transforms and filter
        mass_err = masses * err_ppm / 1e6
        err_abs = mass_err.max()
    err = None
    if USE_MP:
        result = find_xfrms_py(masses, key_mass, err_abs)
        xfrms = result[0]
        err = result[1]
    else:
        xfrms = np.array(find_transforms(masses, key_mass, err_abs, False))

    if len(xfrms) > 0:
        if err is None:
            err = abs((masses[xfrms[:, 2]] - masses[xfrms[:, 1]])
                      - key_mass[xfrms[:, 0]])

        if err_ppm is not None:
            assert isinstance(mass_err, np.ndarray)

            if err_mode == "strict":

                comp_err = mass_err[xfrms[:, 1]]
            elif err_mode == "lax":
                comp_err = mass_err[xfrms[:, 2]]
            else:
                comp_err = np.sqrt(mass_err[xfrms[:, 1]] * mass_err[xfrms[:, 2]])

            ok = abs(err) <= comp_err
            xfrms = xfrms[ok]
            err = err[ok]
        xfrms = pd.DataFrame(xfrms, columns=["xfrm_id", "src", "dst"])
        xfrms["err"] = err
        if clean_peaks:
            xfrms = slowclean(xfrms)
        xfrm_ids = xfrms.values[:, 0]
        edges = xfrms.values[:, 1:3]
    else:
        xfrm_ids = np.array([], dtype=np.int32)
        xfrms = pd.DataFrame(xfrms, columns=["xfrm_id", "src", "dst"])
        edges = []
        err = []
    """
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
    """
    
    edge_data = {"name": xfrm_ids, "xfrm_id": xfrm_ids}
    if key_column is not None:
        edgemap = keys.iloc[xfrm_ids]
        for col in keys.columns:
            edge_data[col] = edgemap[col].values
        
    G = make_graph(masses.shape[0], edges, vdata=vertex_data, edata=edge_data, use_mp=USE_MP, mode=graph_mode)
    G.add_vertex_attrs(name=compound_ids, compound_id=compound_ids)

    
    return xfrms, G

def build_xfrm_graph(xfrms, keys):
    xfrm_ids = xfrms.xfrm_id.values
    edges = xfrms.values[:, 1:3]
    
    edgemap = keys.iloc[xfrm_ids]
    edge_data = {"name": xfrm_ids, "xfrm_id": xfrm_ids}
    for col in keys.columns:
        edge_data[col] = edgemap[col].values

    

def get_intensity_distributions(intensity: pd.DataFrame|np.ndarray):
    pass

    

def get_dataset_transforms(intensity: pd.DataFrame|np.ndarray,
                           compound_info: pd.DataFrame|pd.Series|np.ndarray,
                           keys: pd.DataFrame,
                           err_abs=0.001,
                           err_ppm=1.0,
                           err_mode='moderate',
                           mass_column="mass",
                           key_column='mf',
                           clean_peaks=False,
                           allow_overlap=False,
                           compound_axis=None):
    """Return a dict of sample_id -> raw transform"""
    
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
    result.intensity = intensity
    result.presence = intensity > 0
    if compound_axis == 1:
        result.presence = result.presence.T
        result.intensity = intensity.T

    
    builder = GraphBuilder(keys, key_column, err_abs, err_ppm, err_mode,
                           clean_peaks, allow_overlap)

    mp_args = []

    # Prepare data for processing:
    for samp_id in range(intensity.shape[1-compound_axis]):
        if compound_axis == 0:
            sample = intensity[:, samp_id]
        else:
            sample = intensity[samp_id, ]

        presence = sample > 0            
        samp_mass = masses[presence]
        samp_intens = sample[presence]
        vertex_data = {col: compound_dict[col][presence]
                       for col in compound_dict}

        mp_args.append([samp_mass,
                        compound_id[presence],
                        vertex_data])
                        
    if USE_MP:
        ctx = mp.get_context(MP_MODE)
        with ctx.Pool() as pool:                        
            mp_data = pool.starmap(builder.build_graph, [(x[0], x[1]) for x in mp_args])
        for samp_id, (xfrms, G) in enumerate(mp_data):
            ct_id, ct_num = np.unique(xfrms.xfrm_id, return_counts=True)
            result.counts[ct_id, samp_id] = ct_num
            result.xfrms.append(xfrms)
            result.graphs.append(G)

            
    else:
        for samp_id, args in enumerate(tqdm.tqdm(mp_args, desc="Computing Transforms")):
            xfrms, G = builder.build_graph(*args)            
            ct_id, ct_num = np.unique(xfrms.xfrm_id, return_counts=True)
            result.counts[ct_id, samp_id] = ct_num
            result.xfrms.append(xfrms)
            result.graphs.append(G)
    result.masses = masses
        
    return result


def chain_stats(G, xfrm_ids):
    rows = []
    for xfrm_id in xfrm_ids:
        chain_lens = np.array(G.subgraph_cc_sizes(xfrm_id=xfrm_id))
        nb_p = chain_lens.mean() / chain_lens.var()
        row = {
            "mean_len": chain_lens.mean(),
            "sd_len": chain_lens.std(),
            "negbinom_r": chain_lens.mean() * nb_p / (1-nb_p),
            "nb_p": nb_p
        }
        rows.append(row)
    return pd.DataFrame(rows)
        

    
def transform_stats(data: DataAnalysis):
    assert data.keys is not None

    # First, compute the mean lengths of transformation chain
    mean_lens = np.zeros((data.keys.shape[0], len(data.graphs)))
    mean_lens = np.zeros((data.keys.shape[0], len(data.graphs)))
    chain_counts = np.zeros((data.keys.shape[0], len(data.graphs)))

    for xfrm_id in tqdm.trange(data.keys.shape[0], desc="Commputing chains"):
        for samp_id, G in enumerate(data.graphs):
            # Get the mean chain length for each transformation
            chain_lens = G.subgraph_cc_sizes(xfrm_id=xfrm_id)
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
                            multiples=1,
                           ):
    
    from ._core import find_transforms
    # Ensure transforms are sorted by increasing mass
    xkeys = xkeys.sort_values(mass_col).reset_index(drop=True)

    # Build a transformation network where the nodes and edges come from the
    # xfrm list.

    mass_deltas = xkeys[mass_col].values

    # Look not only for single transforms, but chains of length up to
    # `multiple`
    mass_deltas = np.concat([i * mass_deltas for i in range(1, multiples+1)])
    
    xauto, G = build_sample_graph(xkeys[mass_col],
                                  xkeys._orig_id,
                                  mass_deltas,
                                  key_column=None,
                                  err_abs=err_abs,
                                  err_ppm=None,
                                  graph_mode="ig")

    # A hack to get base igraph object
    G = G.G
    weights = xkeys["_weight"][[(xid % len(xkeys)) for xid in G.es['xfrm_id']]]

    # Find a minimal spanning forest and get the edge ids from it
    # By weighting by mass, we get a small (though perhaps not ideally small)
    # set of transforms. Weighting by inverse count, we get "important"
    # transforms.
    # If we are using multiples, extract the original id (using %)
    min_t = G.spanning_tree(weights=weights)
    min_edges = set([xid % len(xkeys) for xid in min_t.es['xfrm_id']])

    # Now, for each connected component, if *none* of the nodes are in our
    # transform set, we include one node (the one with the lowest
    # weight). Since the component is spanned by the min_tree, this is
    # enough to insure all transforms are reachable.
    comp_xfrms = []
    ccs = G.connected_components()
    print(f"# CCs: {len(ccs)}, sizes = {ccs.sizes()}")
    for cc in ccs:
        # Pick the node with minimum weight. If multiples are being used,
        # prefer shorter multiples
        if not min_edges.intersection(cc):
            node_weights = [(xkeys._weight.loc[node] * (1 + node // len(xkeys)),
                             node % len(xkeys)) for node in cc]
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
                            err_abs=2e-5,
                            multiples=1):
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
        if len(counts.shape) > 1:
            csum = counts.sum(axis=1)
        else:
            csum = counts
        csum = csum / csum.sum()
        xkeys["_count"] = csum
    else:
        xkeys["_count"] = min_thresh
    xkeys["_orig_id"] = xkeys.index

    if weight_by == "mass":
        xkeys['_weight'] = xkeys[mass_col]
        if invert_weights:
            xkeys['_weight'] = 1 / xkeys[mass_col]
    elif weight_by == "count":
        if counts is None:
            raise ValueError("Must supply count matrix when weight_by=='counts'.")
        xkeys['_weight'] =  1 - xkeys._count
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
                                            err_abs=err_abs,
                                            multiples=multiples)
        idx += 1
        print("Iteration ", idx, "   Size=", len(cur_xfrms))
        if len(cur_xfrms) == last_size:
            cur_stable = True
    return cur_xfrms.drop(['_weight', '_count'], axis=1)


def clean_xfrms(da: DataAnalysis):
    """
    Remove all instances where tr<ansforms have the same source or dest
    by picking the one with the lowest error
    """
    out_xfrms = []
    for idx, xfrms in enumerate(da.xfrms):
        xfrms = xfrms.sort_values(["xfrm_id", "dst", "err"]).groupby(["xfrm_id", "dst"]).head(1)
        xfrms = xfrms.sort_values(["xfrm_id", "src", "err"]).groupby(["xfrm_id", "src"]).head(1)
        out_xfrms.append(xfrms)
    result = copy(da)
    result.xfrms = out_xfrms
    return result

def slowclean(xfrms):
    xfrms = xfrms.sort_values(["xfrm_id", "dst", "err"]).groupby(["xfrm_id", "dst"]).head(1)
    xfrms = xfrms.sort_values(["xfrm_id", "src", "err"]).groupby(["xfrm_id", "src"]).head(1)
    return xfrms

def fastclean(xfrms):
    # bah = hardly faster. Boo
    if isinstance(xfrms, pd.DataFrame):
        x = xfrms.values
    else:
        x = xfrms
    # Sort by xfrm_id, dst, err_abs
    ids = np.lexsort([x[:,3], x[:,2], x[:,0]])
    x = x[ids]
    # Find the unique xfrm/dst values
    _, uids = np.unique(x[:, [0,2]], axis=0, return_index=True)
    x = x[uids]

    # Repeat for xfrm_id, src, err_abs
    ids = np.lexsort([x[:,3], x[:,1], x[:,0]])
    x = x[ids]
    # Find the unique xfrm/src values
    _, uids = np.unique(x[:, [0,1]], axis=0, return_index=True)
    x = x[uids]
    return pd.DataFrame(dict(xfrm_id=x[:,0].astype(int),
                             src=x[:,1].astype(int),
                             dst=x[:,2].astype(int),
                             err=x[:,3]))

def findchains(xfrms):
    # This highly unoptimized version outperforms the graph based version
    # by 10x.
    # Of course, it relies on having a "clean" dataset, otherwise it will
    # overcount.
    # This can be detected by looking for seen[j], but what to do in
    # that case? It's ambiguous. Best to break
    result = dict()
    for x_id in xfrms.xfrm_id.unique():
        cycs = []
        sub = xfrms[xfrms.xfrm_id == x_id].values
        d = dict(zip(sub[:,1], np.arange(len(sub))))
        seen = np.zeros(len(sub), dtype=bool)
        for i in range(len(sub)):            
            if  seen[i]:
                continue
            seen[i] = 1
            cnt=0
            dst = int(sub[i][2])
            while dst in d:
                j = d[dst]
                if seen[j]:
                    # We've already counted a chain from this point onward.
                    break
                seen[j] = 1
                dst = int(sub[j][2])
                cnt += 1

            cycs.append(cnt)
        result[x_id] = cycs
    return result



def kendrick(masses, ratio=1.0011178617):
    kmass = masses * ratio
    nom = np.round(kmass)
    kmd = nom - kmass
    return nom, kmd

    
def mass_dist(masses, bin_wid, min_mass=140, max_mass=1200):
    bins = np.arange(min_mass, max_mass + bin_wid, bin_wid) - 0.5
    return np.histogram(masses, bins=bins, density=True)

def intens_dist_rel(intens, bins):
    assert np.all(intens>0)
    # To handle different intensity scales, we'll log normalize the values
    # Relative 
    log_i = np.log10(intens)
    log_i -= log_i.min()
    log_i /= log_i.max()
    return np.histogram(log_i, bins=bins, density=True)


def intens_dist(intens, bins, log10min=5, log10max=9):
    assert np.all(intens>0)
    # To handle different intensity scales, we'll log normalize the values
    # Relative 
    log_i = np.log10(intens)
    bin_edges = np.linspace(log10min, log10max, bins+1)
    return np.histogram(log_i, bins=bin_edges, density=True)


def choose_bins(values_list,
                bin_method="doane",
                summary_method=np.mean,
                use_log=False,
                range=None):
    """
    Given a list of arrays, choose histogram bins suitable for all the arrays.

    Histogram bins are determined using one of the methods of `np.histogram`
    (e.g. "fd", "doane", etc).
    
    If `summary_method` is "all", all the arrays are first combined, and the
    optimal bins for the combined dataset is returned.

    Otherwise, the optimal bins for each dataset are computed, and
    `summary_method` is applied to the data (e.g. min, max, mean, etc)

    NaNs and infs are removed from the data before computation. If
    `use_log` is set, non-positive values are also removed.

    Returns a pair of a range and number of bins.
    """
    # Get rid of NaNs and infs
    values_list = [v[(v > -np.inf) & (v < np.inf)] for v in values_list]
    if use_log:
        values_list = [np.log10(v[v>0]) for v in values_list]
    all_vals = np.concatenate(values_list)
    if range is not None:
        rng = range
    else:
        min_val = np.floor(all_vals.min())
        max_val = np.ceil(all_vals.max())
        rng = (min_val, max_val)
    
    if summary_method == "all":
        # histogram_bin_edges returns the bin edges, so the number of bins is one less
        result = np.histogram_bin_edges(all_vals, range=rng, bins=bin_method).shape[0] - 1
    else:
        bin_sizes = []
        for v in values_list:
            bin_sizes.append(np.histogram_bin_edges(v, range=rng, bins=bin_method).shape[0] - 1)
        result = summary_method(bin_sizes)
    return rng, int(np.ceil(result))

def build_histogram(values_list,
                    bin_method="doane",
                    summary_method=np.mean,
                    use_log=False,
                    range=None):
    values_list = [v[(v > -np.inf) & (v < np.inf)] for v in values_list]
    if use_log:
        values_list = [np.log10(v[v>0]) for v in values_list]
    rng, nbins = choose_bins(values_list,
                             bin_method=bin_method,
                             summary_method=summary_method,
                             range=range)
    # Just get the bin edges. 
    bins = np.linspace(rng[0], rng[1], nbins + 1)
    hists = [np.histogram(v, bins=bins, density=True)[0] for v in values_list]
    result = make_hist_df(hists, bins)
    result.attrs["bin_method"] = bin_method
    result.attrs["bin_summary"] = (
        summary_method if isinstance(summary_method, str)
        else summary_method.__name__)
    return result
    
    

        
    

def choose_mass_intens_bins(da, bin_method='doane', summary_method=np.mean):
    
    masses = []
    specs = []
    
    for idx in range(da.intensity.shape[1]):
        spec = da.intensity[:,idx]
        ms = da.masses[spec>0]
        spec = np.log10(spec[spec>0])
        masses.append(ms)
        specs.append(spec)

    return (choose_bins(masses, bin_method=bin_method, summary_method=summary_method),
            choose_bins(specs, bin_method=bin_method, summary_method=summary_method))
            

        


def make_hist_df(rows, bins, **kwargs):
    result = pd.DataFrame(rows, columns=bins[:-1])
    result.attrs["bin_min"] = bins[0]
    result.attrs["bin_width"] = bins[1] - bins[0]
    result.attrs["bin_num"] = len(bins)-1
    for k, v in kwargs.items():
        result.attrs[k] = v
    return result


def spectral_info(spectrum, masses, G=None, mass_min=150, mass_max=1400):
    KMD_BINS=20
    
    if len(spectrum) != len(masses):
        raise ValueError("Spectrum length much match mass length")
    pres = spectrum >= 0
    masses = masses[pres]
    spectrum = spectrum[pres]
    result ={}

    if G is None:
        xfrm, G = build_sample_graph(masses,
                                     None,
                                     DEFAULT_XFRMS,
                                     "mf",
                                     err_ppm=1.0,
                                     err_mode="moderate",
                                     clean_peaks=True)


    result["mass_hist_1"] = mass_dist(masses, 1, mass_min, mass_max)
    result["mass_hist_10"] = mass_dist(masses, 10, mass_min, mass_max)
    result["mass_hist_50"] = mass_dist(masses, 50, mass_min, mass_max)
    result["intens_abs"] = intens_dist(spectrum, 50)
    result["intens_rel"] = intens_dist_rel(spectrum, 50)
    knom, kmd = kendrick(masses)
    kmass_bins = np.linspace(mass_min, mass_max, KMD_BINS + 1)
    kmd_bins = np.linspace(-0.5, 0.5, KMD_BINS + 1)
    result["kmd_hist"] = np.histogram2d(knom, kmd, bins=[kmass_bins, kmd_bins], density=True)

    result["xfrm_degree"] = np.unique(G.degree(), return_counts=True)
    result["xfrm_between"] =0
    return result

def trim_spectrum(intens, mass):
    mass = mass[intens > 0]
    intens = intens[intens > 0]
    return intens, mass
    
def get_baseline_mass(da, opt_bins, mass_min, mass_max):
    def build_result(ms, bins):
        result = make_hist_df(ms, bins)
        result.attrs["mass_min"] = bins[0] + 0.5
        result.attrs["mass_max"] = bins[-1] - 0.5
        return result 
    m1s = []
    m10s = []
    m50s = []
    mopts = []
    
    result = {}
    for idx in range(da.intensity.shape[1]):
        intens, masses = da.spectrum(idx)
        m1, m1bins = mass_dist(masses, 1, mass_min, mass_max)
        m10, m10bins = mass_dist(masses, 10, mass_min, mass_max)
        m50, m50bins = mass_dist(masses, 50, mass_min, mass_max)
        mopt, moptbins = mass_dist(masses, opt_bins, mass_min, mass_max)
        m1s.append(m1)
        m10s.append(m10)
        m50s.append(m50)
        mopts.append(mopt)

    result["mass_1Da"] = make_hist_df(m1s, m1bins, mass_min=mass_min, mass_max=mass_max)
    result["mass_10Da"] = make_hist_df(m10s, m10bins, mass_min=mass_min, mass_max=mass_max)
    result["mass_50Da"] = make_hist_df(m50s, m50bins, mass_min=mass_min, mass_max=mass_max)
    result["mass_auto"] = make_hist_df(mopts, moptbins, mass_min=mass_min, mass_max=mass_max)
    return result
    

def get_baseline_xfrms(da, bin_method, summary_method):
    degrees = []
    betweens = []
    closes = []
    dists = []
    for idx in range(len(da.graphs)):
        G = da.graphs[idx]
        # For the quantities with integral values, we compute the count for
        # each value here.
        degrees.append(G.degrees())

        dists.append(G.distances())


        betweens.append(np.array(G.betweenness()))
        closes.append(np.array(G.closeness()))

    result = {}

    maxdeg = max([max(d) for d in degrees])
    maxdist = max([max(d) for d in dists])

    # Make sure we have all degrees/distances up to and including max, and
    # replace missing values with 0
    result["degrees"] = make_hist_df(degrees, np.arange(maxdeg+1)).replace(np.nan, 0)
    result["distances"] = make_hist_df(dists, np.arange(-1, maxdist+1)).replace(np.nan, 0)

    max_between = np.ceil(max([max(b) for b in betweens]))

    # Determine best number of bins for data and build histograms
    result["betweenness"] = build_histogram(betweens,
                                            use_log=True,
                                            bin_method=bin_method,
                                            summary_method=summary_method)
    result["closeness"] = build_histogram(closes,
                                            use_log=False,
                                            bin_method=bin_method,
                                            summary_method=summary_method)

    return result

        
# For finding the distance between distributions, we will use the "Earth Movers
# Distance". We have generally simple cases (one-D distributions with equal
# weights), so in this case it can be proved that the solution is just the
# absolute difference of the cumulativs sums

def baseline_distances(hists):

    # This just uses some numpy indexing tricks to speed multiple computations
    wid = hists.attrs["bin_width"]
    num_rows = hists.shape[0]
    cumsums = np.array([np.cumsum(row) for _, row in hists.iterrows()])

    # Find indices of all unique pairs of samples
    X, Y = np.triu_indices(num_rows, k=1)
    dists = np.abs(cumsums[X] - cumsums[Y]).sum(axis=1)
    return dists * wid / hists.attrs["bin_num"]

def bad_sliced_wasserstein(X, Y, num_proj):
    import scipy.stats as stats
    '''Takes:
        X: 2d (or nd) histogram
        Y: 2d (or nd) histogram
        num_proj: Number of random projections to compute the mean over
        ---
        returns:
        mean_emd_dist'''
    """
     NOTE: BAD - DO NOT USE!
     This is not a "sliced wasserstein". True sliced wasserstein works by projecting
    the 2D PMFs/histograms onto all lines in 2D space (through the origin or
    through the PMF center),  computing the 1-D wasserstein_distance for each
    projection, and taking the mean. For any projection, the total probability
    mass projected on the line is equal to the total probability of all bins. Thus,
    if X and Y have equal mass, each projection will too, and thus the wasserstein
    distance is well defined.
    One key property is that the true sliced wasserstein distance is invariant to transpose (or rotation)

    This random code from the internet does not project down to a line. Instead,
    it takes a weighted mean of the columns, and computes the 1-D wasserstein_distance
    of each, taking the mean. Note in particular, that the "projections" of X and
    Y may have different sums, so the wasserstein_distance is not as easy to
    compute. Furthermore, this method is NOT invariant to transpose -
    bad_sliced_wasserstein(X, Y) != bad_sliced_wasserstein(X.T, Y.T)
    
    """
    #% Implementation of the (non-generalized) sliced wasserstein (EMD)
    # for 2d distributions as described here: https://arxiv.org/abs/1902.00434 %#
    # X and Y should be a 2d histogram
    # Code adapted from stackoverflow user:
    # Dougal - https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
    dim = X.shape[1]
    ests = []
    if num_proj is None:
        dirs = np.eye(dim)
    else:
        dirs = np.random.rand(dim, num_proj)
        dirs /= np.linalg.norm(dirs, axis=0)
        dirs = dirs.T

    for dir in dirs:

        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(stats.wasserstein_distance(np.arange(dim), np.arange(dim), X_proj, Y_proj))
    return np.mean(ests)


def baseline_distances_2d(prepped, khist):
    """
    For pre-prepared 2D histograms, return the pairwise distances between
    them.

    Here, "prepare" means that these are the results of "prep_sliced_wasserstein";
    That is, the inputs are the cumulative sums (column-wise) of radon transforms
    of the histograms.

    """
    N = len(prepped)

    # Like all wasserstein_distances, the maximum of the raw distances
    # between cululative sums is given by the number of bins. For a 2D
    # histogram, it will be the maximum dimension minus 1
    max_col = max(khist.attrs["bins_x"], khist.attrs["bins_y"]) - 1
            

    x_dim, y_dim = prepped[0].shape
    assert(all(p.shape == (x_dim, y_dim) for p in prepped))
    
    X, Y = np.triu_indices(N, k=1)
    dists = [compute_sliced_wasserstein(prepped[i], prepped[j])
             for i,j in zip(X,Y)]    
    return np.array(dists) / max_col

def cross_distances_2d(prep1, prep2, khist):
    max_col = max(khist.attrs["bins_x"], khist.attrs["bins_y"]) - 1

    num_1 = len(prep1)
    num_2 = len(prep2)
    # Generate indices for all pairwise comparisons
    I1 = np.tile(np.arange(num_1), num_2)
    I2 = np.repeat(np.arange(num_2), num_1)
    dists = [compute_sliced_wasserstein(prep1[i], prep2[j])
             for i,j in zip(I1, I2)]
    return pd.DataFrame(np.array(dists).reshape((num_2, num_1)) / max_col)


def cross_distances(hist1, hist2):
    """
    Compute the distances between histograms in hist1 to those in hist2.
    The inputs are DataFrames, where the rows are samples and the columns
    histogram bins.
    Distances are computed via a simplified Wasserstein distance, which assumes all
    histograms integrate to 1.
    """

    if ((hist1.shape[1] != hist2.shape[1]) or
        not np.allclose(hist1.columns,hist2.columns)):
        raise ValueError("Trying to compare histograms with different bins")

    assert np.allclose(hist1.sum(axis=1)* hist1.attrs['bin_width'], 1.0)
    assert np.allclose(hist2.sum(axis=1) * hist2.attrs['bin_width'], 1.0)
    assert np.allclose(hist1.sum(axis=1)[0], hist2.sum(axis=1)[0])

    
    # This just uses some numpy indexing tricks to speed multiple computations
    wid = hist1.attrs["bin_width"]
    nbins = hist1.attrs["bin_num"]
    num_1 = hist1.shape[0]
    num_2 = hist2.shape[0]
    cumsums1 = np.array([np.cumsum(row) for _, row in hist1.iterrows()])
    cumsums2 = np.array([np.cumsum(row) for _, row in hist2.iterrows()])
    # Generate indices for all pairwise comparisons
    I1 = np.tile(np.arange(num_1), num_2)
    I2 = np.repeat(np.arange(num_2), num_1)
    

    dists = np.abs(cumsums1[I1] - cumsums2[I2]).sum(axis=1)
    # Now, reshape so that the rows are the samples of hist2,
    # and columns are the samples of hist1
    dists = dists.reshape((num_2, num_1))
    
    return pd.DataFrame(dists * wid / nbins)


MASS_MATS = [
    "mass_1Da",
    "mass_10Da",
    "mass_50Da",
    "mass_opt",
]

NORM_MATS = [
    "mass_1Da",
    "mass_10Da",
    "mass_50Da",
    "mass_opt",
    "intensities_rel",
    "intensities_abs",
    "betweenness",
    "closeness",    
    ]

ALL_2D_MATS = [
    "kendrick",
    ]

ALL_1D_MATS = NORM_MATS + [
    "degrees",
    "distances",
    "xfrm_counts",
    ]

ALL_MATS = ALL_1D_MATS + ALL_2D_MATS

FINAL_MATS = [
    "mass_opt",
    "kendrick",
    "intensities_rel",
    "betweenness",
    "closeness",
    "degrees",
    "distances",
    "xfrm_counts"]


FINAL_METRICS = {
    "spectral": ["mass_opt",
                 "kendrick",
                 "intensities_rel"],
    "network": ["betweenness",
                "closeness",
                "degrees",
                "distances",
                "xfrm_counts"]
}

"""
Note:
So the issue with storing histograms for the baseline is that the input data
may not be aligned with the bins. In particular, its range might extend above or below
the histogram.

In that case, we need to preserve the existing bins, but add extra bins in front of or behind.

To align, we need all the values from the test spectra (either singular or a dataanalysis)

* Step 1: get range of values
* 2: Compute extra pre/post bins
* 3: Add 0 columns to baseline matrix
"""

def align_histogram(hist, test_values):
    hist_min = hist.attrs["bin_min"]
    hist_wid = hist.attrs["bin_width"]
    # This is the max *right* side of the last bin
    hist_max = hist_min + hist_wid * (hist.attrs["bin_num"])

    if isinstance(test_values, list):
        test_values = np.concat(test_values)
    test_min = np.min(test_values)
    test_max = np.max(test_values)

    num_low = int(max(0, np.ceil((hist_min - test_min) / hist_wid)))
    num_hi = int(max(0, np.ceil((test_max-hist_max) / hist_wid)))

    if (num_low == 0 and num_hi == 0):
        # Already aligned
        return hist
    
    ndat = np.pad(hist.values, ((0, 0), (num_low, num_hi)))
    new_min = hist_min - num_low * hist_wid
    new_max = hist_max + num_hi * hist_wid
    new_nbin = hist.shape[1] + num_low + num_hi
    # The columns denote the left side of each bin, so we need to subtract
    # off the bin width.
    new_cols = np.linspace(new_min, new_max - hist_wid, new_nbin)
    result = pd.DataFrame(ndat, columns=new_cols)
    result.attrs.update(hist.attrs)
    result.attrs["bin_min"] = new_min
    result.attrs["bin_num"] = new_nbin
    return result

def histogram_like(hist, test_values):
    """
    Given an existing histogram and a set of values,
    align the histogram to the values (by adding empty bins on either side)
    and then create a new histogram of the values with the same bins.
    Returns a pair of histograms (aligned original, new)
    """
    hist_min = hist.attrs["bin_min"]
    hist_wid = hist.attrs["bin_width"]
    # This is the max *right* side of the last bin
    hist_max = hist_min + hist_wid * (hist.attrs["bin_num"])

    all_values = np.concat(test_values)
    test_min = np.min(all_values)
    test_max = np.max(all_values)

    # How much padding do we need on either side??
    num_low = int(max(0, np.ceil((hist_min - test_min) / hist_wid)))
    num_hi = int(max(0, np.ceil((test_max-hist_max) / hist_wid)))

    new_min = hist_min - num_low * hist_wid
    new_max = hist_max + num_hi * hist_wid
    new_nbin = hist.shape[1] + num_low + num_hi
    new_bins = np.linspace(new_min, new_max, new_nbin + 1)

    if (num_low > 0 or num_hi > 0):
        ndat = np.pad(hist.values, ((0, 0), (num_low, num_hi)))
        # The columns denote the left side of each bin, so we need to subtract
        # off the bin width.
        new_hist = pd.DataFrame(ndat, columns=new_bins[:-1])
        new_hist.attrs.update(hist.attrs)
        hist = new_hist
        hist.attrs["bin_min"] = new_min
        hist.attrs["bin_num"] = new_nbin

    # Now, create similarly binned histograms for the test data:
    test_hist = build_histogram(test_values, bin_method=new_nbin, range=(new_min, new_max))    
    
    return hist, test_hist
    
def unit_histogram_like(hist1, hist2):
    hist1 = hist1.copy()
    hist2 = hist2.copy()
    all_cols = set(hist2.columns).union(hist1.columns)
    
    h1 = pd.DataFrame(columns=sorted(list(all_cols)))
    h1[list(all_cols.intersection(hist1.columns))] = hist1
    h1[list(all_cols.difference(hist1.columns))] = 0

    h2 = pd.DataFrame(columns=sorted(list(all_cols)))
    h2[list(all_cols.intersection(hist2.columns))] = hist2
    h2[list(all_cols.difference(hist2.columns))] = 0
    h1.attrs["bin_width"] = 1
    h1.attrs["bin_min"] = min(all_cols)
    h1.attrs["bin_num"] = len(all_cols)
    h2.attrs.update(hist1.attrs)
    return h1, h2
    
            


def compute_test_stats(test_da):
    def make_unit_hist(datadict):
        df = pd.DataFrame(datadict).replace(np.nan, 0)
        df = df.div(df.sum(axis=1), axis=0)
        return df
    
    test_stats = defaultdict(list)
    for idx in range(test_da.intensity.shape[1]):
        ins, ms = test_da.spectrum(idx)
        test_stats["masses"].append(ms)
        ivals = np.log10(ins)
        irel = ivals - ivals.min()
        irel /= ivals.max()
        test_stats["intensities_abs"].append(ivals)
        test_stats["intensities_rel"].append(irel)

    test_stats["tot_xfrm_counts"] = test_da.counts.sum(axis=0)
    # Store the relative number of each transform. Transpose it so it
    # is sample per row.
    test_stats["xfrm_counts"] = pd.DataFrame((test_da.counts / test_da.counts.sum(axis=0)).T)

    if USE_MP:
        ctx = mp.get_context(MP_MODE)
        with ctx.Pool() as pool:                        
            mp_data = pool.map(TransformGraph.call_stats, test_da.graphs)
        for d in mp_data:
            close = d['clos']
            test_stats["closeness"].append(close[close>-np.inf])
            btwn = d['btwn']
            test_stats["betweenness"].append(np.log10(btwn[btwn>0]))
            test_stats["distances"].append(d['dists'])
            test_stats["degrees"].append(d['degs'])
            
    else:        
        for g in test_da.graphs:
            close = g.closeness()
            test_stats["closeness"].append(close[close>-np.inf])
            btwn = g.betweenness()
            test_stats["betweenness"].append(np.log10(btwn[btwn>0]))
            test_stats["distances"].append(g.distances())
            test_stats["degrees"].append(g.degrees())

    # For stats with integral values (degrees, distances), we'll store as
    # a histogram
    test_stats["distances"] = make_unit_hist(test_stats["distances"])
    test_stats["degrees"] = make_unit_hist(test_stats["degrees"])

    return test_stats


KMD_NOM_BINS=50
KMD_DEFECT_BINS = 20

def baseline_kendrick(masses, mass_min, mass_max):
    kmass_bins = np.linspace(mass_min, mass_max, KMD_NOM_BINS + 1)
    kmd_bins = np.linspace(-0.5, 0.5, KMD_DEFECT_BINS + 1)
    hists = []
    for ms in masses:
        knom, kmd = kendrick(ms)
        hist, X, Y = np.histogram2d(knom, kmd, bins=[kmass_bins, kmd_bins], density=True)
        hists.append(hist.flatten())

    X, Y = np.meshgrid(X[:-1], Y[:-1])
    X = np.round(X.flatten(), 1)
    Y = np.round(Y.flatten(), 2)
    result = pd.DataFrame(np.array(hists), columns = zip(X, Y))
    result.attrs["mass_min"] = mass_min
    result.attrs["mass_max"] = mass_max
    result.attrs["bins_x"] = KMD_NOM_BINS
    result.attrs["bins_y"] = KMD_DEFECT_BINS
    result.attrs["bin_wid_x"] = (mass_max - mass_min)/KMD_NOM_BINS
    result.attrs["bin_wid_y"] = 1/KMD_DEFECT_BINS
    result.attrs["bin_area"] = result.attrs["bin_wid_x"] * result.attrs["bin_wid_y"]
    
    return result

def prep_sliced_wasserstein(hists):
    from skimage.transform import radon
    nx = hists.attrs["bins_x"]
    ny = hists.attrs["bins_y"]
    results = []
    for idx, hist in hists.iterrows():
        dat = hist.values.reshape(nx, ny)
        # Take the radon transform to project onto lines through the center.
        # SKimage will pad the image so that it lies in a center of a circle
        # with 'circle=False'
        # We'll use the default number of angles (180), though we probably
        # could do fewer
        rad = radon(dat, circle=False)
        # Now, for some interpolation reason, the column sums are not exactly
        # equal as they should be (they are close). Normalize them all to 1
        rad /= rad.sum(axis=0)
        # Finally, we might as well prepare the column-wuse cumulative sums
        results.append(rad.cumsum(axis=0))
    return results

def compute_sliced_wasserstein(prepX, prepY):
    return np.abs(prepX - prepY).sum(axis=0).mean()
        
def sliced_wasserstein(X, Y):
    from skimage.transform import radon
    assert X.shape == Y.shape
    rx = radon(X, circle=False)
    rx /= rx.sum(axis=0)
    ry = radon(Y, circle=False)
    ry /= ry.sum(axis=0)

    return np.abs(rx.cumsum(axis=0) - ry.cumsum(axis=0)).sum(axis=0).mean()
    

    
def build_baseline(test_stats, bin_method="doane", summary_method="all"):
    def augment_unit_histogram(df):
        df.attrs["bin_min"] = 0
        df.attrs["bin_width"] = 1
        df.attrs["bin_num"] = df.shape[1]
        return df

    import scipy.stats as stats
    
    binmeths = dict(bin_method=bin_method, summary_method=summary_method)

    result = {}
    
    ((mass_min, mass_max), mass_bins) = choose_bins(test_stats["masses"], **binmeths)
    ((int_min, int_max), int_bins) = choose_bins(test_stats["intensities_abs"], **binmeths)

    result["mass_opt"] = build_histogram(test_stats["masses"], **binmeths)
    result["mass_1Da"] = build_histogram(test_stats["masses"],
                                         bin_method=int(mass_max-mass_min))
    result["mass_10Da"] = build_histogram(test_stats["masses"],
                                          bin_method=int(mass_max-mass_min) // 10)
    result["mass_50Da"] = build_histogram(test_stats["masses"],
                                          bin_method=int(mass_max-mass_min) // 50)
    result["intensities_abs"] = build_histogram(test_stats["intensities_abs"], **binmeths)
    result["intensities_rel"] = build_histogram(test_stats["intensities_rel"], **binmeths)

    result["kendrick"] = baseline_kendrick(test_stats["masses"], mass_min, mass_max)

    result["betweenness"] = build_histogram(test_stats["betweenness"], **binmeths)
    result["closeness"] = build_histogram(test_stats["closeness"], **binmeths)


    # The following are already stored as histograms
    result["degrees"] = augment_unit_histogram(test_stats["degrees"])
    result["distances"] = augment_unit_histogram(test_stats["distances"])

    # For the counts, we'll do things a little differently.
    # We already have a histogram of sorts, but we'll arrange the transforms
    # in decreasing order of occurrence.
    xcount_order = np.argsort(test_stats["xfrm_counts"].sum(axis=0))[::-1]
    df = test_stats["xfrm_counts"][xcount_order]
    result["xfrm_counts"] = augment_unit_histogram(df)
    result["xfrm_totals"] = test_stats["tot_xfrm_counts"]
    

    result["kdes"] = {}
    # Compute within baseline test_stats
    for mat_name in ALL_1D_MATS:
        mat = result[mat_name]
        dists = baseline_distances(mat)
        mat.attrs["baseline_min_dist"] = dists.min()
        mat.attrs["baseline_max_dist"] = dists.max()
        mat.attrs["baseline_mean_dist"] = dists.mean()
        mat.attrs["baseline_var_dist"] = dists.var()
        mat.attrs["baseline_median_dist"] = np.median(dists)
        result["kdes"][mat_name] = stats.gaussian_kde(dists)

    result["2d_prep"] = {}
    for mat_name in ALL_2D_MATS:
        mat = result[mat_name]        
        prepped = prep_sliced_wasserstein(result[mat_name])
        result["2d_prep"][mat_name] = prepped
        dists = baseline_distances_2d(prepped, mat)

        mat.attrs["baseline_min_dist"] = dists.min()
        mat.attrs["baseline_max_dist"] = dists.max()
        mat.attrs["baseline_mean_dist"] = dists.mean()
        mat.attrs["baseline_var_dist"] = dists.var()
        mat.attrs["baseline_median_dist"] = np.median(dists)
        result["kdes"][mat_name] = stats.gaussian_kde(dists)
    return result

def plot_baseline(baseline, axs=None, **plotargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    n_plots = len(baseline["kdes"])
    if axs is None:
        fig, axs = plt.subplots(nrows=int(np.ceil(n_plots / 4)), ncols=4, **plotargs)
        sns.despine()
        fig.tight_layout()
    
    for idx, (name, kde) in enumerate(baseline["kdes"].items()):
        ax = axs[idx // 4][idx % 4]
        X = np.linspace(kde.dataset.min(), kde.dataset.max())
        Y = kde(X)
        ax.plot(X, Y)
        ax.set_title(name, fontsize=9)
        
    return axs


def plot_comp(baseline, comp, axs=None, **plotargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    n_plots = len(baseline["kdes"])
    if axs is None:
        fig, axs = plt.subplots(nrows=int(np.ceil(n_plots / 4)), ncols=4, **plotargs)
        sns.despine()
        fig.tight_layout()
    
    for idx, (name, mat) in enumerate(baseline["kdes"].items()):
        kde = ST.gaussian_kde(comp[name].values.flatten())
        
        ax = axs[idx // 4][idx % 4]
        X = np.linspace(kde.dataset.min(), kde.dataset.max())
        Y = kde(X)
        ax.plot(X, Y)
        ax.set_title(name, fontsize=9)
        
    return axs

def root_mean_square(arr, axis=None):
    return np.sqrt(np.square(arr).mean(axis=axis))

def ECDF(sorted_dat, test_dat):
    """Return the empirical CDF applied to test_dat.
    We could use scipy.stats.ecdf, but if the empirical data is already sorted
    this is much faster."""
    return np.searchsorted(sorted_dat, test_dat) / len(sorted_dat)

    
def plain_z_score(bl, test_dists, key):
    bl_dat = bl.raw_dists[key]
    mean = np.mean(bl_dat)
    return (test_dists - mean) / bl_dat.std()

def robust_z_score(bl, test_dists, key):
    bl_dat = bl.raw_dists[key]
    median = np.median(bl_dat)
    MAD = ST.median_abs_deviation(bl_dat)
    return np.abs(test_dists - median) / (1.4826 * MAD)

def quantile_score(bl, test_dists, key):
    """
    Quantile normalization approach. For each test sample, we have the distances
    between it and all baseline samples. Find the percentile of each distance
    (compared to inter-baseline distances).
    """
    bl_dat = bl.raw_dists[key]
    cdf = ST.ecdf(bl_dat).cdf
    return cdf.evaluate(test_dists)


def compare_spectra(test_stats, baseline, metrics=None):
    if metrics is None:
        metrics = FINAL_METRICS

    # First, compute all the pairwise distances between all samples in test_stats
    # and all samples in the baseline.
    # The errors are all normalized between 0 and 1

    result = {}
    all_masses = np.concatenate(test_stats["masses"])
    for key in MASS_MATS:
        bl_hist, test_hist = histogram_like(baseline[key], test_stats["masses"])

        result[key] = cross_distances(bl_hist, test_hist)

    for key in ["intensities_abs",
                "intensities_rel",
                "betweenness",
                "closeness"]:
        bl_hist, test_hist = histogram_like(baseline[key], test_stats[key])
        result[key] = cross_distances(bl_hist, test_hist)
        

    for key in ["degrees",
                "distances"]:
        bl_df, ts_df = unit_histogram_like(baseline[key], test_stats[key])
        result[key] = cross_distances(bl_df, ts_df)

    # For transform counts, we need to rearrange columns:
    bl_xc = baseline["xfrm_counts"]
    bl_xc, ts_xc = unit_histogram_like(bl_xc, test_stats["xfrm_counts"][bl_xc.columns])


    result["xfrm_counts"] = cross_distances(bl_xc, ts_xc)

    kmin = baseline["kendrick"].attrs["mass_min"]
    kmax = baseline["kendrick"].attrs["mass_max"]
    ts_kmd = baseline_kendrick(test_stats["masses"], kmin, kmax)
    ts_prep = prep_sliced_wasserstein(ts_kmd)
    result["kendrick"] = cross_distances_2d(baseline["2d_prep"]["kendrick"],
                                            ts_prep, ts_kmd)


    """
    # Now, using all the computed distances, find the distance scores for each
    # submetric for each sample
    quantiles = {}
    robust_zs = {}
    submetrics = sum(metrics.values(), [])
    for mat in submetrics:
        robust_zs[mat] = np.median(robust_z_score(baseline, result, mat), axis=1)
        
        quantiles[mat] = np.median(
            2*(quantile_score(baseline, result, mat) - 0.5), axis=1)

    robust_zs = pd.DataFrame(robust_zs)
    quantiles = pd.DataFrame(quantiles)
    for level in metrics:
        submetrics = metrics[level]
        # For robust_z, take square mean
        robust_zs[level] = root_mean_square(robust_zs[submetrics], axis=1)
        quantiles[level] = root_mean_square(quantiles[submetrics], axis=1)
    submetrics = list(metrics.keys())
    robust_zs["overall"] = root_mean_square(robust_zs[submetrics], axis=1)
    quantiles["overall"] = root_mean_square(quantiles[submetrics], axis=1)

        
    result["robust_z"] = robust_zs
    result["quantiles"] = quantiles
    """

    return result


def mpl_args(kws, **defaults):
    """
    Utility for handling plotting arguments.
    Overrides any values in defaults with values from `kws` and returns
    kws. Handles case when kws is None
    For example, to override a linestyle, instead of:
    >>> ax.plot(X, Y, linestyle=":", **plotargs)
    use:
    >> ax.plot(X, Y, **mpl_args(plot_args, linestyle=":"))
    """
    result = defaults
    if kws is None:
        kws = {}
    result.update(kws)
    return result

def gen_axes_grid(names, n_cols, axs=None, fig_kw=None, title=True, tight=True, **title_args):
    """
    Generate a 2D grid of axes corresponding to the elements of `names`.
    Also generates titles for each subplot from `names`,
    If `names` is a dict, uses the values for the titles.
    If `axs` is provided, determines number of rows/columns
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    if isinstance(names, dict):
        titles = list(names.values())
    else:
        titles = names
    
    n_rows = int(np.ceil(len(names) / n_cols))

    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, **mpl_args(fig_kw))
        if tight:
            fig.tight_layout()
        if isinstance(axs[0], Axes):
            axs = [axs]
    else:
        n_rows = len(axs)
        if isinstance(axs[0], Axes):
            n_cols = len(axs)
            n_rows = 1
            axs = [axs]
        n_cols = len(axs[0])
        fig = axs[0][0].get_figure()
        assert n_rows * n_cols >= len(names)

    # Get rid of extra axes:
    for idx in range(len(names), n_rows * n_cols):
        ax = axs[idx // n_cols][idx % n_cols]
        ax.set_visible(False)

    if title:
        for idx, title in enumerate(titles):
            ax = axs[idx // n_cols][idx % n_cols]
            ax.set_title(title, **title_args)
    return fig, axs, n_cols

            
class Baseline:

    def __init__(self,
                 intensity: pd.DataFrame|np.ndarray,
                 mass_info: pd.DataFrame|pd.Series|np.ndarray,
                 transforms: pd.DataFrame,
                 err_abs=0.001,
                 err_ppm=1.0,
                 err_mode='moderate',
                 mass_column="mass",
                 key_column='mf',
                 clean_peaks=False,
                 allow_overlap=False,
                 compound_axis=None,
                 bin_method="doane",
                 summary_method="all",
                 provenance=None):

        self.transforms = transforms
        self.key_column = key_column
        self.err_abs = err_abs
        self.err_ppm = err_ppm
        self.err_mode = err_mode
        self.clean_peaks = clean_peaks
        self.allow_overlap = allow_overlap
        self.bin_method = bin_method
        self.summary_method = summary_method

        self.provenance = provenance
        # Masses and intensities stored in DataAnalysis,
        # so we don't save them directly.
        self.data_analysis = get_dataset_transforms(intensity,
                                                    mass_info,
                                                    transforms,
                                                    err_abs,
                                                    err_ppm,
                                                    err_mode,
                                                    mass_column,
                                                    key_column,
                                                    clean_peaks,
                                                    allow_overlap,
                                                    compound_axis)

        self.test_stats = compute_test_stats(self.data_analysis)
        self.histos = {}
        # For a given metric, all pairwise distances, sorted
        self.raw_dists = {}

        # For 2-D histograms, prepare the radon transforms for later
        # comparison
        self.radons = {}

        self._build_baseline()

    def save(self, fname):
        import pickle
        pickle.dump(self, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        import pickle
        return pickle.load(open(fname, 'rb'))

    def kde(self, mat_name):
        return stats.gaussian_kde(self.raw_dists[mat_name])


    def prepare_comp(self, test_intensity, test_mass_info, mass_column='mf',
                     compound_axis=None):

        test_da = get_dataset_transforms(test_intensity,
                                         test_mass_info,
                                         self.transforms,
                                         self.err_abs,
                                         self.err_ppm,
                                         self.err_mode,
                                         mass_column,
                                         self.key_column,
                                         self.clean_peaks,
                                         self.allow_overlap,
                                         compound_axis)
        test_stats = compute_test_stats(test_da)
        return (test_stats, test_da)

    def compare_prepped(self, test_stats, test_da=None, final_metrics=None):
        #
        return SpectralComparison(self, test_stats, test_da=test_da,metrics=final_metrics)

    


    def compare_spectra(self, test_intensity, test_mass_info, mass_column='mf',
                        compound_axis=None, final_metrics=None):

        test_da = get_dataset_transforms(test_intensity,
                                         test_mass_info,
                                         self.transforms,
                                         self.err_abs,
                                         self.err_ppm,
                                         self.err_mode,
                                         mass_column,
                                         self.key_column,
                                         self.clean_peaks,
                                         self.allow_overlap,
                                         compound_axis)
        test_stats = compute_test_stats(test_da)
        return SpectralComparison(self, test_stats, test_da=test_da,metrics=final_metrics)



    
        

    def _build_baseline(self):
        def augment_unit_histogram(df):
            df.attrs["bin_min"] = 0
            df.attrs["bin_width"] = 1
            df.attrs["bin_num"] = df.shape[1]
            return df

        import scipy.stats as stats
        test_stats = self.test_stats

        binmeths = dict(bin_method=self.bin_method, summary_method=self.summary_method)

        ((mass_min, mass_max), mass_bins) = choose_bins(test_stats["masses"], **binmeths)
        ((int_min, int_max), int_bins) = choose_bins(test_stats["intensities_abs"], **binmeths)

        self.histos["mass_opt"] = build_histogram(test_stats["masses"], **binmeths)
        self.histos["mass_1Da"] = build_histogram(test_stats["masses"],
                                             bin_method=int(mass_max-mass_min))
        self.histos["mass_10Da"] = build_histogram(test_stats["masses"],
                                              bin_method=int(mass_max-mass_min) // 10)
        self.histos["mass_50Da"] = build_histogram(test_stats["masses"],
                                              bin_method=int(mass_max-mass_min) // 50)
        self.histos["intensities_abs"] = build_histogram(test_stats["intensities_abs"], **binmeths)
        self.histos["intensities_rel"] = build_histogram(test_stats["intensities_rel"], **binmeths)

        self.histos["kendrick"] = baseline_kendrick(test_stats["masses"], mass_min, mass_max)

        self.histos["betweenness"] = build_histogram(test_stats["betweenness"], **binmeths)
        self.histos["closeness"] = build_histogram(test_stats["closeness"], **binmeths)


        # The following are already stored as histograms
        self.histos["degrees"] = augment_unit_histogram(test_stats["degrees"])
        self.histos["distances"] = augment_unit_histogram(test_stats["distances"])

        # For the counts, we'll do things a little differently.
        # We already have a histogram of sorts, but we'll arrange the transforms
        # in decreasing order of occurrence.
        xcount_order = np.argsort(test_stats["xfrm_counts"].sum(axis=0))[::-1]
        df = test_stats["xfrm_counts"][xcount_order]
        self.histos["xfrm_counts"] = augment_unit_histogram(df)
        self.histos["xfrm_totals"] = test_stats["tot_xfrm_counts"]


        # Compute within baseline test_stats
        for mat_name in ALL_1D_MATS:
            mat = self.histos[mat_name]
            dists = baseline_distances(mat)
            mat.attrs["baseline_min_dist"] = dists.min()
            mat.attrs["baseline_max_dist"] = dists.max()
            mat.attrs["baseline_mean_dist"] = dists.mean()
            mat.attrs["baseline_var_dist"] = dists.var()
            mat.attrs["baseline_median_dist"] = np.median(dists)
            self.raw_dists[mat_name] = np.sort(dists)

        for mat_name in ALL_2D_MATS:
            mat = self.histos[mat_name]        
            prepped = prep_sliced_wasserstein(self.histos[mat_name])
            self.radons[mat_name] = prepped
            dists = baseline_distances_2d(prepped, mat)

            mat.attrs["baseline_min_dist"] = dists.min()
            mat.attrs["baseline_max_dist"] = dists.max()
            mat.attrs["baseline_mean_dist"] = dists.mean()
            mat.attrs["baseline_var_dist"] = dists.var()
            mat.attrs["baseline_median_dist"] = np.median(dists)
            self.raw_dists[mat_name] = np.sort(dists)

    def plot_baseline(self, axs=None, fig_kw=None, which=None, **plotargs):
        import seaborn as sns
        if which is None:
            which = self.raw_dists.keys()
        fig, axs, n_cols = gen_axes_grid(which, n_cols=4, axs=axs, fig_kw=fig_kw)
        sns.despine()

        for idx, name in enumerate(which):
            dists = self.raw_dists[name]
            kde = ST.gaussian_kde(dists)
            ax = axs[idx // n_cols][idx % n_cols]
            X = np.linspace(kde.dataset.min(), kde.dataset.max())
            Y = kde(X)
            ax.plot(X, Y, **mpl_args(plotargs, linestyle=":", color="black"))

        return axs

def gen_massim_files(baseline: Baseline,
                     dirname: str,
                     prefix: str|None = None,
                     xfrm_err: float = 1e-5,
                     xfrm_multiples: int = 2,
                     intensity_bins: int = 100):
    import os
    import os.path
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    if prefix is None:
        prefix = ""
    elif not prefix.endswith("_"):
        prefix = prefix + "_"
    # First, the masses of the input data, sorted by count.

    masses = baseline.data_analysis.masses
    assert baseline.data_analysis.presence is not None
    assert baseline.data_analysis.intensity is not None
    assert masses is not None

    mass_df = pd.DataFrame(
        {"mass": masses,
         "count": baseline.data_analysis.presence.sum(axis=1)})
    # Next, a list of isolated masses (those with no transform connection)
    iso = np.unique(
        np.concat([g.isolated_compounds()
                   for g in baseline.data_analysis.graphs]))
    mass_df["isolated"] = False
    mass_df.loc[iso, "isolated"] = True
    mass_df = mass_df.sort_values("count", ascending=False).reset_index(drop=True)

    mass_df.to_csv(os.path.join(dirname, f"{prefix}masses.csv"))
        

    # Get transform probabilities:
    xfrm_stats = transform_stats(baseline.data_analysis)
    xfrm_stats.to_csv(os.path.join(dirname, f"{prefix}transform_stats.csv"))
    

    # Repeat for nonredundant transforms
    keys_nr = find_nonredundant_xfrms(baseline.transforms,
                                      mass_col=baseline.key_column,
                                      counts=baseline.data_analysis.counts,
                                      err_abs=xfrm_err,
                                      weight_by='count',
                                      multiples=xfrm_multiples)
    # Find stats for nonredundant
    da_nr = get_dataset_transforms(baseline.data_analysis.intensity,
                                   masses,
                                   keys=keys_nr,
                                   key_column=baseline.key_column,
                                   err_mode=baseline.err_mode,
                                   err_ppm=baseline.err_ppm,
                                   allow_overlap=baseline.allow_overlap,
                                   clean_peaks=baseline.clean_peaks)
    xfrm_stats_nr = transform_stats(da_nr)
    xfrm_stats_nr.to_csv(os.path.join(dirname, f"{prefix}transform_stats_nr.csv"))

    # Find empirical distribution for baseline:

    qs = np.linspace(0, 1, intensity_bins)
    icdfs = [np.quantile(intens, qs, method="inverted_cdf")
             for intens in baseline.test_stats["intensities_abs"]]
    icdfs = pd.DataFrame(icdfs, columns=qs)
    icdfs.to_csv(os.path.join(dirname, f"{prefix}intensity_distributions.csv"))

    
        
        
        


        

class SpectralComparison:
    def __init__(self, baseline, test_stats, test_da=None, metrics=None):
        if metrics is None:
            metrics = FINAL_METRICS

        self.baseline = baseline
        self.test_stats = test_stats
        self.test_da = test_da

        self.distances = {}
        self.test_histos = {}
        self._compute_distances()

        self.metrics = {}


        # Now, using all the computed distances, find the distance scores for each
        # submetric for each sample
        quantiles = {}
        robust_zs = {}
        closest_z = {}
        submetrics = sum(metrics.values(), [])
        for mat in submetrics:
            robust_zs[mat] = robust_z_score(baseline,
                                            np.median(self.distances[mat], axis=1),
                                            mat)

            quantiles[mat] = quantile_score(baseline,
                                            np.median(self.distances[mat], axis=1),
                                            mat)

        robust_zs = pd.DataFrame(robust_zs)
        quantiles = pd.DataFrame(quantiles)
        for level in metrics:
            submetrics = metrics[level]
            # For robust_z, take square mean
            robust_zs[level] = np.mean(robust_zs[submetrics], axis=1)
            quantiles[level] = np.mean(quantiles[submetrics], axis=1)
        submetrics = list(metrics.keys())
        robust_zs["overall"] = np.mean(robust_zs[submetrics], axis=1)
        quantiles["overall"] = np.mean(quantiles[submetrics], axis=1)

        self.summary_z = pd.DataFrame(columns=robust_zs.columns)
        self.summary_z.loc["Median"] = robust_zs.median()
        self.summary_z.loc["IQR"] = ST.iqr(robust_zs, axis=0)
        self.summary_z.loc["Pr(>2)"] = (robust_zs > 2).sum() / robust_zs.shape[0]
        self.summary_z = self.summary_z.transpose()
        
        self.metrics["robust_z"] = robust_zs
        self.metrics["quantiles"] = quantiles

    def _compute_distances(self):
        # Compute all the pairwise distances between all samples in test_stats
        # and all samples in the baseline.
        # The errors are all normalized between 0 and 1
        test_stats = self.test_stats
        baseline = self.baseline.histos
        all_masses = np.concatenate(test_stats["masses"])
        for key in MASS_MATS:
            bl_hist, test_hist = histogram_like(baseline[key], test_stats["masses"])

            self.distances[key] = cross_distances(bl_hist, test_hist)
            self.test_histos[key] = test_hist

        for key in ["intensities_abs",
                    "intensities_rel",
                    "betweenness",
                    "closeness"]:
            bl_hist, test_hist = histogram_like(baseline[key], test_stats[key])
            self.distances[key] = cross_distances(bl_hist, test_hist)
            self.test_histos[key] = test_hist


        for key in ["degrees",
                    "distances"]:
            bl_df, ts_df = unit_histogram_like(baseline[key], test_stats[key])
            self.distances[key] = cross_distances(bl_df, ts_df)
            self.test_histos[key] = ts_df


        # For transform counts, we need to rearrange columns:
        bl_xc = baseline["xfrm_counts"]
        col_order = bl_xc.columns
        bl_xc, ts_xc = unit_histogram_like(bl_xc, test_stats["xfrm_counts"][bl_xc.columns])
        bl_xc.columns = col_order
        ts_xc.columns = col_order
        self.test_histos["xfrm_counts"] = ts_xc


        self.distances["xfrm_counts"] = cross_distances(bl_xc, ts_xc)

        kmin = baseline["kendrick"].attrs["mass_min"]
        kmax = baseline["kendrick"].attrs["mass_max"]
        ts_kmd = baseline_kendrick(test_stats["masses"], kmin, kmax)
        ts_prep = prep_sliced_wasserstein(ts_kmd)
        raw_kendrick = cross_distances_2d(self.baseline.radons["kendrick"],
                                          ts_prep, ts_kmd)
        self.distances["kendrick"] = raw_kendrick
        self.test_histos["kendrick"] = ts_kmd

        
    def plot_comp(self, which=None, axs=None, fig_kw=None, **plotargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        if which is None:
            which = self.baseline.raw_dists.keys()
        fig, axs, n_cols = gen_axes_grid(which, n_cols=4, axs=axs, fig_kw=fig_kw)

        self.baseline.plot_baseline(which=which, axs=axs)
        for idx, name in enumerate(which):
            mat = self.distances[name]
            kde = ST.gaussian_kde(mat.values.flatten())

            ax = axs[idx // n_cols][idx % n_cols]
            X = np.linspace(kde.dataset.min(), kde.dataset.max())
            Y = kde(X)
            ax.plot(X, Y, **mpl_args(plotargs))

        return axs

    def compare_hists(self, hist_name, ax=None, alpha="auto", fig_kw=None, **plotargs):
        import matplotlib.pyplot as plt

        test_hists = self.test_histos[hist_name]
        bl_hists = self.baseline.histos[hist_name]

        wid = bl_hists.attrs["bin_width"] / 2

        if ax is None:
            _, ax = plt.subplots(**mpl_args(fig_kw))

        # Special case for xfrm_counts, as it has reordered columns:
        if hist_name == "xfrm_counts":
            
            test_hists = test_hists.copy()
            test_hists.columns = range(test_hists.shape[1])
            bl_hists = bl_hists.copy()
            bl_hists.columns = range(bl_hists.shape[1])

        if alpha == "auto":
            t_alpha = 1 / test_hists.shape[0]
            b_alpha = 1 / bl_hists.shape[0]
        else:
            t_alpha = alpha
            b_alpha = alpha
        for ix in range(bl_hists.shape[0]):
            
            ax.bar(bl_hists.columns, bl_hists.iloc[ix], alpha=b_alpha,
                   color='black', width=wid, **plotargs)

        for ix in range(test_hists.shape[0]):
            ax.bar(test_hists.columns + wid, test_hists.iloc[ix], alpha=t_alpha,
                   color='red', width=wid, **plotargs)
        ax.set_title(hist_name, fontsize=9)
        return ax

    def compare_all_hists(self, which=None, axs=None, alpha="auto", fig_kw=None, **plotargs):
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        if which is None:
            which = ["mass_opt",
                     "intensities_abs",
                     "betweenness",
                     "closeness",
                     "degrees",
                     "distances",
                     "xfrm_counts"]
            
        fig, axs, n_cols = gen_axes_grid(which, n_cols=4, axs=axs, fig_kw=fig_kw)

        for idx, name in enumerate(which):
            ax = axs[idx // n_cols][idx % n_cols]
            self.compare_hists(name, ax=ax, alpha=alpha, **plotargs)
        return axs


DIST_WHICH = {
    "mass_opt": "Mass",
    "intensities_abs": "Intensity",
    "kendrick": "Kendrick Mass Defect",
    "degrees": "Graph Degree",
    "betweenness": "Graph Betweenness",
    "closeness": "Graph Closeness",
    "distances": "Graph Distances",
    "xfrm_counts": "Transformation Counts",
    }

HIST_WHICH = {
    "mass_opt": "Mass",
    "intensities_abs": "Intensity",
    "degrees": "Graph Degree",
    "betweenness": "Graph Betweenness",
    "closeness": "Graph Closeness",
    "distances": "Graph Distances",
    "xfrm_counts": "Transformation Counts",
    }

    

    

    

    
     

    
