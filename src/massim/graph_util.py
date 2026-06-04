import numpy as np
import pandas as pd

try:
    import igraph as ig
    HAS_IG = True
except ImportError:
    HAS_IG = False

"""
# Graph-tool is faster, but it doesn't play well with either python's
# multiprocessing library, or my OpenMP code. Probably, the correct
# link arguments might fix this, but I can't figure them out.
try:
    import graph_tool.all as gt
    HAS_GT = True
except ImportError:
    HAS_GT = False
"""
HAS_GT = False

def make_graph(vcount, edges, vdata=None, edata=None, mode='auto', use_mp=False):
    if vdata is None:
        vdata = {}
    if edata is None:
        edata = {}
    if HAS_GT and mode in ["gt", "auto"]:
        if not use_mp:
            return GraphGT(vcount, edges, vdata, edata)
    if HAS_IG and mode in ['ig', "auto"]:
        return GraphIgraph(vcount, edges, vdata, edata)



class TransformGraph:
    @staticmethod
    def call_betweenness(g):
        return g.betweenness()
    @staticmethod
    def call_closeness(g):
        return g.closeness()
    @staticmethod
    def call_distances(g):
        return g.distances()

    @staticmethod
    def call_stats(g):
        return dict(
            ccs=g.subgraph_cc_sizes(),
            dists=g.distances(),
            btwn=g.betweenness(),
            clos=g.closeness(),
            degs=g.degrees(),
        )
    

    
    
    def subgraph_cc_sizes(self, **edge_filter):
        pass
    def betweenness(self):
        pass
    def closeness(self, harmonic=True):
        pass
    def distances(self):
        pass
    def degrees(self):
        pass


    

class GraphIgraph(TransformGraph):
    def __init__(self, vcount, edges, vdata, edata):
        super().__init__()
        self.G = ig.Graph(vcount, edges)
        for vattr in vdata:
            self.G.vs[vattr] = vdata[vattr]
        for eattr in edata:
            self.G.es[eattr] = edata[eattr]

    def add_edge_attrs(self, **kwargs):
        for eattr in kwargs:
            self.G.es[eattr] = kwargs[eattr]

    def add_vertex_attrs(self, **kwargs):
        for vattr in kwargs:
            self.G.vs[vattr] = kwargs[vattr]


    def subgraph_cc_sizes(self, **edge_filter):
        """
        Filter the graph and  a list of the sizes of the connected components.
        Only return component sizes with at least one edge.
        """
        esub = self.G.subgraph_edges(self.G.es.select(**edge_filter))
        return np.array(esub.connected_components().sizes())

    def betweenness(self):
        return np.array(self.G.betweenness())

    def closeness(self, harmonic=True):
        if harmonic:
            return np.array(self.G.harmonic_centrality())
        else:
            return np.array(self.G.closeness())


    def distances(self):
        dist = np.array(self.G.distances())
        dist = dist[np.triu_indices_from(dist, k=1)]
        dist = dist[dist != np.inf]
        dist, dist_count = np.histogram(dist, bins=np.arange(1, dist.max()+2))
        return dict((int(a), int(b)) for a,b in zip(dist_count, dist))

    def degrees(self):
        return dict((int(b[0]), b[2]) for b in self.G.degree_distribution().bins())

    def isolated_compounds(self):
        return np.array(
            [x[0] for x in zip(self.G.vs()["compound_id"], self.G.degree())
             if x[1] == 0])

    


class GraphGT(TransformGraph):
    def typematch(self, typ):
        match typ:
            case np.int32: return "int"
            case np.int64: return "long"
            case np.uint32: return "unsigned int"
            case np.uint64: return "unsigned long"
            case np.float32: return "float"
            case np.float64: return "double"
            case np.bool: return "bool"
            case "int": return "long"
            case "bool": return "bool"
            case "float": return "float64"
            case "str": return "string"
            case _:
                raise ValueError(f"Unknown type: {typ}")
        

    def prop_type(self, vals):
        if isinstance(vals[0], str):
            return "string"
        if isinstance(vals, np.ndarray):
            return self.typematch(vals.dtype)
        return self.typematch(type(vals[0]).__name__)
        

    def add_prop(self, name, values, is_edge):
        ptype = self.prop_type(values)
        if ptype == "string":
            # Unneccessary for now
            return
        if is_edge:
            prop = self.G.new_edge_property(ptype)
        else:
            prop = self.G.new_vertex_property(ptype)
        prop.get_array()[:] = values
        if is_edge:
            self.G.ep[name] = prop
        else:
            self.G.vp[name] = prop

            
    def __init__(self, vcount, edges, vdata, edata):
        super().__init__()
        self.G = gt.Graph(directed=False)
        self.G.add_vertex(vcount)
        self.G.add_edge_list(edges)

        self.add_vertex_attrs(**vdata)
        self.add_edge_attrs(**edata)

    def add_edge_attrs(self, **kwargs):
        for eattr in kwargs:
            self.add_prop(eattr, kwargs[eattr], is_edge=True)

    def add_vertex_attrs(self, **kwargs):
        for vattr in kwargs:
            self.add_prop(vattr, kwargs[vattr], is_edge=False)


    def subgraph_cc_sizes(self, **edge_filter):
        """
        Filter the graph and  a list of the sizes of the connected components.
        Only return component sizes with at least one edge.
        """
        edge_filter = list(edge_filter.items())
        if len(edge_filter) != 1:
            raise NotImplementedError("Only a single filter at a time with graph_tool")
        edge_filter = edge_filter[0]
        eprop = self.G.edge_properties[edge_filter[0]]
        esub = gt.GraphView(self.G, efilt=eprop.a==edge_filter[1])

        _, hist = gt.label_components(esub)
        return hist[hist>1]

    def betweenness(self, harmonic=True):
        a,b = gt.betweenness(self.G, norm=False)
        return np.array(a)

    def closeness(self, harmonic=True):
        return  np.array(gt.closeness(self.G, norm=True, harmonic=harmonic))

    def distances(self):
        a, b = gt.distance_histogram(self.G)
        # For undirected graphs, we divide by 2 to get unique paths
        # We also don't need the "0" length entry, as its always zero
        return dict((int(a), int(b)) for a,b in zip(b[1:-1], a[1:]/2))

    def degrees(self):
        cnt, bins = gt.vertex_hist(self.G, 'total')
        return dict((int(a), int(b)) for a,b in zip(bins[:-1], cnt))
