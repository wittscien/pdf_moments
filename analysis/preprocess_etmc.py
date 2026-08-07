import numpy as np

import funcs as tp


def preprocess(params, relist, data2, data3):
    # All hadrons use the same configurations and the same resampling list.
    for k in data2.keys():
        assert len(data2[k]) == params['N'], 'Run ETMC 2pt with read2=direct again.'

    redata2 = {}
    for k in data2.keys():
        redata2[k] = tp.resample(data2[k], params['tech'], relist)

    # Fold the 2pt correlator around t=0.
    for k in redata2.keys():
        redata2[k] = (np.roll(redata2[k][:,::-1], 1, axis=1) + redata2[k]) / 2

    redata3 = {}
    for k in data3.keys():
        redata3[k] = {}
        for dt in data3[k].keys():
            assert len(data3[k][dt]) == params['N'], 'Run ETMC 3pt with read3=direct again.'
            redata3[k][dt] = tp.resample_general(data3[k][dt], params['tech'], relist)

    return redata2, redata3
