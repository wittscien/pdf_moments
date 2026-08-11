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
import reconstruction

#%%
if __name__ == '__main__':
    ensembles = ['cA211', 'cB211', 'cC211']
    tech = 'bootstrap'

    result_root = Path('../mydata/main/spectra')
    metadata_root = Path('../mydata/main/corr')
    figure_root = Path('../figures/extrapolation')
    output_root = Path('../mydata/main/extrapolation')

    #%%
    output_root.mkdir(parents=True, exist_ok=True)
    [params, metadata, data] = reading.reading(ensembles, tech, result_root, metadata_root)

    #%%
    continuum_correlated = True
    continuum = extrapolate.continuum_extrapolation(params, metadata, data, ensembles, fitting_ranges.ranges_continuum, continuum_correlated, figure_root)

    #%%
    flow_correlated = False
    limits = extrapolate.flow_extrapolation(params, metadata, data, continuum, ensembles, fitting_ranges.ranges_flow, flow_correlated, figure_root)

    #%%
    # Temporarily use the pion priors for the pion and both kaon flavor insertions.
    pdf_prior = True
    pion_pdf = reconstruction.pdf_reconstruction('pion', limits['pion'], pdf_prior, figure_root)
    kaon_pdf = reconstruction.pdf_reconstruction('kaon', limits['kaon'], pdf_prior, figure_root)
    kaon_s_pdf = reconstruction.pdf_reconstruction('kaon_s', limits['kaon_s'], pdf_prior, figure_root)

    #%%
    # with open(output_root / 'continuum_bootstrap.pckl', 'wb') as dfile: pickle.dump(continuum,dfile)
    # with open(output_root / 'flow_limits_bootstrap.pckl', 'wb') as dfile: pickle.dump(limits,dfile)
    # with open(output_root / 'pion_pdf_bootstrap.pckl', 'wb') as dfile: pickle.dump(pion_pdf,dfile)
