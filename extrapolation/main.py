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
import ratio

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
    params_by_ensemble, metadata_by_ensemble, result_by_ensemble = reading.read_results(ensembles, tech, result_root, metadata_root)

    #%%
    selected_by_ensemble = {}
    selected_tsep_by_ensemble = {}
    for ensemble in ensembles:
        selected_by_ensemble[ensemble], selected_tsep_by_ensemble[ensemble] = ratio.largest_tsep_result(result_by_ensemble[ensemble], params_by_ensemble[ensemble].get('tsep_list'))
        ratio.plot_ratio_vs_tf(selected_by_ensemble[ensemble], metadata_by_ensemble[ensemble], tech, ensemble, figure_root, xaxis='tf')

    #%%
    continuum = extrapolate.continuum_extrapolation(selected_by_ensemble, ensembles, params_by_ensemble, metadata_by_ensemble, tech, fitting_ranges.ranges_continuum, figure_root)

    #%%
    limits = extrapolate.flow_extrapolation(continuum, tech, fitting_ranges.ranges_flow, figure_root)

    #%%
    with open(output_root / ('continuum_%s.pckl' % tech), 'wb') as output: pickle.dump(continuum, output)
    with open(output_root / ('flow_limits_%s.pckl' % tech), 'wb') as output: pickle.dump(limits, output)
    with open(output_root / ('largest_tsep_%s.pckl' % tech), 'wb') as output: pickle.dump(selected_tsep_by_ensemble, output)
    with open(output_root / ('largest_tsep_ratios_%s.pckl' % tech), 'wb') as output: pickle.dump(selected_by_ensemble, output)
