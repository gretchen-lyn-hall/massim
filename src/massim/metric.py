import numpy as np

def dstats(df, transpose=None):
    if hasattr(df, "df"):
        df = df.df
    if transpose is None:
        if df.shape[0] > df.shape[1]:
            transpose = True
    if transpose:
        df = df.transpose()

    pres = (df > 0).sum(axis=0) 
    pres /= pres.max()
    return pres

def presence_histo(df, transpose=None, bins=50, minval=2):
    if transpose is None:
        if df.shape[0] > df.shape[1]:
            transpose = True
    if transpose:
        df = df.transpose()

    pres = (df > 0).sum(axis=0) - minval
    pres /= pres.max()
    result =  np.histogram(pres, bins=bins, range=(0, 1), density=True)
    result = (result[0] / bins, result[1])
    return result
    
def earth_mover_dist(h1, h2):
    assert (h1[0].shape[0] == h2[0].shape[0])
    assert np.all(h1[1] == h2[1])
    wid = h1[1][1] - h1[1][0]
    emd = np.zeros(h1[0].shape[0] + 1)
    for i in range(len(h1[0])):
        emd[i+1] = h1[0][i] + emd[i] - h2[0][i]
    return np.abs(emd).sum()
        
def plotres(results, src_hists, plt):
    hs = [presence_histo(res.df) for res in results]
    for src in src_hists:
        dists = [earth_mover_dist(h, src) for h in hs]
        plt.plot(dists)
