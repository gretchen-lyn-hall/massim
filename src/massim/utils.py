from collections import OrderedDict, defaultdict
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class PrettyDict(OrderedDict):
    __setattr__ = OrderedDict.__setitem__
    __getattr__ = OrderedDict.__getitem__

    def __init__(self):
        super().__init__()
        super().__setattr__("_format", defaultdict(list))
                            
        
    def __repr__(self):
        lines = []
        float_format = "{}"
        cur_indent = 0;
        for idx, (k, v) in enumerate(self.items()):
            for fmt in  self._format.get(idx, []):
                if fmt[0] == "spacer":
                    lines.append("")
                elif fmt[0] == "float_fmt":
                    float_format = f"{{:{fmt[1]}}}"
                elif fmt[0] == "indent":
                    cur_indent += fmt[1]
                elif fmt[0] == "header":
                    lines.append(fmt[1])
                else:
                    raise AssertionError(f"Unknown format code '{fmt[0]}'")
            if isinstance(v, (float, np.floating)):
                v = float_format.format(v)
            lines.append(" " * cur_indent + f"{k}: {v}")
            
        return '\n'.join(lines)

    def add_spacer(self):
        self._format[len(self)].append(("spacer",))

    def add_header(self, text):
        self._format[len(self)].append(("header", text))

    def indent(self, indent=4):
        self._format[len(self)].append(("indent", indent))

    def dedent(self, indent=4):
        self._format[len(self)].append(("indent", -indent))


    def set_format(self, fmt):
        self._format[len(self)].append(("float_fmt", fmt))

    def set_round(self, digits):
        self.set_format(f".{digits}")


def sort_diffs_naive(X):
    diffs = X - X[:, None]
    XY = np.indices((len(X), len(X)))
    
    # Filter out mass differences by range, and flatten the arrays
    valid_locs = np.where(diffs > 0, True, False)
    diffs = diffs[valid_locs]
    XY = XY[:, valid_locs].transpose()
    reord = diffs.argsort(kind='stable')
    return (diffs[reord], XY[reord])
    

def sort_diffs(X):
    from collections import deque
    N = len(X)
    N2 = N * (N-1) // 2
    X = np.sort(X)
    deltas = X[1:] = X[:-1]
    dsort = np.argsort(deltas)
    result_diff = np.zeros(N2)
    result_coord = np.zeros((N2, 2), dtype = np.int32)

    # Prime queue
    result_diff[0] = deltas[dsort[0]]
    result_coord[0] = (dsort[0], dsort[0]+1)
    result_diff[1] = deltas[dsort[1]]
    result_coord[0] = (dsort[1], dsort[1]+1)
    # Bah, this won't work.
    

    
    
    
    
