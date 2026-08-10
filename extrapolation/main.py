#!/opt/anaconda3/bin/python3
# Author: Haobo Yan
#%%
import pickle
import sys
from pathlib import Path

sys.path.append('../analysis')

import extrapolate
import fitting_ranges
import reading

#%%
if __name__ == '__main__':
    ensembles = ['cA211', 'cB211', 'cC211']
    tech = 'jackknife'

    result_root = Path('../mydata/main/spectra')
    metadata_root = Path('../mydata/main/corr')
    figure_root = Path('../figures/extrapolation')
    output_root = Path('../mydata/main/extrapolation')

    #%%
    output_root.mkdir(parents=True, exist_ok=True)
    [params, metadata, data] = reading.reading(ensembles, tech, result_root, metadata_root)

    #%%
    continuum = extrapolate.continuum_extrapolation(params, metadata, data, ensembles, fitting_ranges.ranges_continuum, figure_root)

    #%%
    limits = extrapolate.flow_extrapolation(continuum, fitting_ranges.ranges_flow, figure_root)

    #%%
    with open(output_root / 'continuum_bootstrap.pckl', 'wb') as dfile: pickle.dump(continuum,dfile)
    with open(output_root / 'flow_limits_bootstrap.pckl', 'wb') as dfile: pickle.dump(limits,dfile)
