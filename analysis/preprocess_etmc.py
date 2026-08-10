import numpy as np

import funcs as tp


def preprocess(params, relist, data2, data3):
    # Do binning, resampling, folding, and shifting
    # All hadrons use the same configurations, bin boundaries, and resampling list.
    for k in data2.keys():
        assert len(data2[k]) == params['N'], 'Run ETMC 2pt with read2=direct again.'
    for k in data3.keys():
        for dt in data3[k].keys():
            assert len(data3[k][dt]) == params['N'], 'Run ETMC 3pt with read3=direct again.'

    # Bin the data
    if params['bin']:
        bindata2 = {}
        bindata3 = {}
        for k in data2.keys():
            bindata2[k] = tp.bin_data(data2[k], params['Nb'])
        for k in data3.keys():
            bindata3[k] = {}
            for dt in data3[k].keys():
                bindata3[k][dt] = tp.bin_data(data3[k][dt], params['Nb'])
    else:
        bindata2 = data2
        bindata3 = data3

    redata2 = {}
    for k in bindata2.keys():
        redata2[k] = tp.resample(bindata2[k], params['tech'], relist)

    # Fold the 2pt correlator around t=0.
    for k in redata2.keys():
        redata2[k] = (np.roll(redata2[k][:,::-1], 1, axis=1) + redata2[k]) / 2

    redata3 = {}
    for k in bindata3.keys():
        redata3[k] = {}
        for dt in bindata3[k].keys():
            redata3[k][dt] = tp.resample_general(bindata3[k][dt], params['tech'], relist)

    return redata2, redata3
