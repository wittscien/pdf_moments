from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs
import fitting_ranges
from flow_matching import c_numeric



def fit_three(params, xR, R, metadata, result):
    tau = params['tau']
    T = params['T']
    relen = params['relen']

    fit_range = fitting_ranges.ranges_three[params['ensemble']]

    three_name = '%s'%(params['ensemble'])
    three_tit = r'$Ens=%s$'%(params['ensemble'])
    three_dir = '%s'%(params['ensemble'])

    nflow = len(metadata['tau_list']) - 1

    result_3pt_chi2dof = {}
    result_3pt = {}
    selected_3pt = {}
    for k in R.keys():
        print(k)

        result_3pt_chi2dof[k] = {}
        result_3pt[k] = {}
        selected_3pt[k] = {}

        if k == 'pion':
            k_one = 'pion'
        elif k in ['kaon', 'kaon_s']:
            k_one = 'kaon'
        tit = r'$Ens=%s \quad %s$' % (params['ensemble'], k)
        for n in range(3, 7):
            result_3pt_chi2dof[k][n] = {}
            result_3pt[k][n] = {}
            selected_3pt[k][n] = {}
            for tf in range(nflow + 1):
                result_3pt_chi2dof[k][n][tf] = {}
                result_3pt[k][n][tf] = {}
                selected_3pt[k][n][tf] = {}
                for itsep, tsep in enumerate(params['tsep_list']):
                    # R[k][n][tf][tsep][ls]
                    middle = tsep // 2

                    # Lazy priors
                    prior = {}
                    prior['C'] = R[k][n][tf][tsep][0][middle]
                    params['mtype'] = mtype = 'const'
                    params['prior'] = [prior['C']]

                    # selected
                    params['tins'] = {1:{}}
                    for k in R.keys():
                        selected_3pt['n'][k][n][tf][tsep], selected_3pt['tins'][k][n][tf][tsep] = fit_range[k][n][tf][tsep]
                        params['tins'][1][k] = [0, T//2]

                    # corr fit
                    R_onlyone = {}
                    R_onlyone[k] = np.copy([k][n][tf])
                    [selected_lazy_tins,result_3pt_chi2dof[k][n][tf][tsep],result_3pt[k][n][tf][tsep]] = tp.fitting_3pt(prior,params,R_onlyone,mtype,selected_3pt,correlated=True)

                    if params['lazy_tins']:
                        selected_3pt['tins'][k][n][tf][tsep] = selected_lazy_tins

                # Plot stability
                tp.plot_stability_3pt(k,params,selected_3pt,result_3pt_chi2dof[k][n][tf][tsep],result_3pt[k][n][tf][tsep],tit=r'$Ens=%s %s$'%(params['ensemble'],inputs.labels(k)),sv='%s_%s_n%d_tf%d'%(three_name,k,n,tf),figdir='fit_three/%s'%(three_dir))

                # Plot results
                tp.plot_result_3pt(R_onlyone,params,selected_3pt,result_3pt_chi2dof[k][n][tf][tsep],result_3pt[k][n][tf][tsep],tau=tau,mtype=params['mtype'],tit=r'$Ens=%s %s$'%(params['ensemble'],inputs.labels(k)),sv='%s_%s_n%d_tf%d'%(three_name,k,n,tf),figdir='fit_three/%s'%(three_dir))

    return result_3pt_chi2dof, result_3pt
