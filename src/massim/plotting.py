from .experiment import ExperimentResult

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button


def plot_spectrum(res: ExperimentResult, samp_id: int,  ax=None, log=False):
    masses = res.species_info.mass
    intens = res.abundance[samp_id, :]
    if log:
        intens = np.log10(intens)
    if ax is None:
        fig, ax = plt.subplots()
    segs =  [([x, 0], [x,y]) for x,y in zip(masses, intens)]
    coll = LineCollection(segs, linewidths=1)
    ax.add_collection(coll)
    ax.set_ylim(0, intens.max())
    ax.set_xlim(masses.min(), masses.max())
    return ax



def plot_spectra(res: ExperimentResult, log=False, auto_lim=True):
    fig, ax = plt.subplots()
    masses = res.species_info.mass

    class Index:
        ind = 0
        cur_col = None
        lim_set = False
        def redraw(self):
            if self.cur_col is not None:
                self.cur_col.remove()
            intens = res.abundance[self.ind, :]
            segs =  [([x, 0], [x,y]) for x,y in zip(masses, intens)]
            self.cur_col = LineCollection(segs, linewidths=0.5)
            ax.add_collection(self.cur_col)
            coord_labels = ", ".join(res.sample_coords.columns)
            coord_text = ", ".join(str(x) for x in res.sample_coords.iloc[self.ind])
            ax.set_title(f"Sample #{self.ind} ('{res.sample_coords.index[self.ind]}' "
                     f"  {{{coord_labels}}} = ({coord_text}))")
            if auto_lim or not self.lim_set:
                if isinstance(auto_lim, float):
                    ymax = auto_lim
                else:
                    ymax = intens.max()
                ax.set_ylim(0, ymax)
                ax.set_xlim(masses.min(), masses.max())
                self.lim_set = True
            fig.canvas.draw()
            fig.canvas.flush_events()

            
        def next(self, event):
            print("NXT")
            if self.ind < res.abundance.shape[0] - 1:
                self.ind += 1
            self.redraw()
        def prev(self, event):
            print("PRV")
            if self.ind > 0:
                self.ind -= 1
            self.redraw()
    callback = Index()
    callback.redraw()
    axprev = fig.add_axes((0.7, 0.05, 0.1, 0.075))
    axnext = fig.add_axes((0.81, 0.05, 0.1, 0.075))
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    plt.show()
            

def intensity_hist(res: ExperimentResult, samp_id: int,  ax=None, log=False, bins=100):
    intens = res.abundance[samp_id, :]
    intens = intens[intens > 0]
    if log:
        intens = np.log10(intens)
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(intens, bins=bins)
    return ax
