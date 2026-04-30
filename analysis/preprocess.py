import os
import numpy as np
from scipy import linalg
import funcs as tp
import inputs



def filename(npt, mydatadir, ensemble):
    file = '../%s/corr/%s/data_%s_%s.pckl'%(mydatadir, ensemble, npt, ensemble)
    refile = '../%s/corr/%s/redata_%s_%s.pckl'%(mydatadir, ensemble, npt, ensemble)
    return file, refile

def should_update(reference_file, target_file):
    if not os.path.exists(target_file):
        return True
    else:
        return os.path.getmtime(target_file) < os.path.getmtime(reference_file)

def preprocess(args, datadir, params, relist, data2, data3):
    # Do binning, resampling, folding, and shifting
    # Bin the data
    relen = relist.shape[0]
    mydatadir = datadir['mydata']
    ensemble = params['ensemble']
    fold = True

    redata2 = {}
    redata2_Nsrc_sink = {}
    redata3 = {}
    redata3_conserved = {}
    redata4 = {}

    # 2pt
    if args.read2 != 'no':
        [file, refile] = filename('2pt', mydatadir, ensemble)
        if should_update(file, refile):
            print('Resampling 2pt...', flush=True)
            bindata2 = data2
            redata2 = {}
            for k in bindata2.keys():
                redata2[k] = tp.resample(bindata2[k], params['tech'], relist)
            if fold:
                for k in redata2.keys():
                    if k.startswith('pion'):
                        redata2[k] = (np.roll(redata2[k][:,::-1],1,axis=1)+redata2[k])/2
            tp.write_data(refile, redata2)
        else:
            redata2 = tp.fast_read(refile)

    # 3pt
    if args.read3 != 'no':
        [file, refile] = filename('3pt', mydatadir, ensemble)
        if should_update(file, refile):
            print('Resampling 3pt...', flush=True)
            bindata3 = data3
            redata3 = {}
            for k in bindata3.keys():
                redata3[k] = tp.resample_general(bindata3[k], params['tech'], relist)
            tp.write_data(refile, redata3)
        else:
            redata3 = tp.fast_read(refile)

    return redata2, redata3
