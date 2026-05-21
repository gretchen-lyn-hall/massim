import igraph as ig
import scipy.sparse as SP
import numpy as np

from .distributions import (
    RNG, gen_param_dist, Distribution, UniformDistribution, NormalDistribution, ConstantDistribution
)


def test_graph():
    g = ig.Graph.Full(3)
    g.vs()["frac"] = [.1, .25, .15]
    g.es()["frac"] = [.25, .1, .4]
    return GraphSim(g)
    

class GraphSim:
    def __init__(self, xfrm_graph):
        self.g = xfrm_graph
        self.Nv = self.g.vcount()
        self.Ne = self.g.ecount()
        
        adj = ig.adjacency._get_adjacency_sparse(self.g)
        self.adj = adj
        self.nbrs = adj.sum(axis=0)
        X, Y = adj.nonzero()
        self.n_edge = len(X) // 2
        self.e_prob = np.array(self.g.es()["frac"])
        self.e_prob /= self.e_prob.sum()

        # we also add a "supersource, connected to all vertices.
        
        self.v_prob = np.array(self.g.vs()["frac"])
        self.v_prob /= self.v_prob.sum()

        R = np.arange(self.Nv)
        # Indices for the supersource, which will be the last row/column
        A = R
        B = np.repeat((self.Nv), self.Nv)

        upp = X > Y
        # So, the given adjacency graph is symmetric, but the indices are not in anyway
        # particular order. Rearrange indices into upper, then lower, then
        # supersource, then diagonal
        self.X = np.concatenate((X[upp], Y[upp], A, B, R))
        self.Y = np.concatenate((Y[upp], X[upp], B, A, R))
        self.R = R
        self.supvals = np.concatenate((self.v_prob, 1-self.v_prob))
        
        # Get compounds with no transitions:
        z_idx = (adj.sum(axis=0) == 0).A1
        self.loners = np.arange(self.Nv)[z_idx]

        

    def gen_mat(self, ewts):
        assert len(ewts) == self.n_edge
        # Scale edge weights to 0-1
        ewts = (np.tanh(ewts) + 1)/2

        # Initially create net with edge probs and supersource values,
        # but zeros on diagonal
        wts = np.concat((ewts, 1-ewts, self.supvals, np.zeros(self.Nv)))
        result =  SP.csr_matrix((wts, (self.X, self.Y)), shape=(self.Nv+1, self.Nv+1))
        # Set diagonal to be the remainder of the outgoing probability for each
        # edge - that is, for each outgoing 'p', add '1-p'
        # Include the supersource
        #result[self.R, self.R] = (self.nbrs+1) - result.sum(axis=0)[0,:-1]
        return result / result.sum(axis=0)

        
        
class SpeciesGen:
    def __init__(self,
                 gs: GraphSim,
                 n_species: int,
                 edge_wt_dist: Distribution = None,
                 comp_wt_dist: Distribution = None,
                 edge_footprint_var: float = 0,
                 comp_footprint_var: float = 0,
                 overlap: int=10):
        
        self.gs = gs
        self.n_spcs = n_species
        self.e_avg = overlap * gs.Ne / n_species
        self.c_avg = overlap * gs.Nv / n_species

        if edge_wt_dist is None:
            edge_wt_dist = UniformDistribution(-1, 1)
        self.edge_wt_dist = edge_wt_dist
        

        if edge_footprint_var == 0:
            self.edge_foot = ConstantDistribution(self.e_avg)
        else:
            self.edge_foot = NormalDistribution(self.e_avg,
                                                self.e_avg * edge_footprint_var)

        if comp_footprint_var == 0:
            self.comp_foot = ConstantDistribution(self.c_avg)
        else:
            self.comp_foot = NormalDistribution(self.c_avg,
                                                self.c_avg * comp_footprint_var)

        # We have N species. Each is responsible for e_avg of the edges

    def sample(self, rng=None):
        rng = RNG(rng)

        counts = self.edge_foot(self.n_spcs, rng=rng).astype(int)
        idxs = [rng.choice(self.gs.n_edge, p=self.gs.e_prob, size=c, replace=False)
                for c in counts]
        wts = [self.edge_wt_dist(c) for c in counts]
        return idxs,wts
        
        
        
