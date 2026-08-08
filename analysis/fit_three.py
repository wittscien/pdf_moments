import numpy as np
import funcs as tp
import inputs
import fitting_ranges


def fit_three(params, xR, R, metadata, result):
    fit_range = fitting_ranges.ranges_three[params['ensemble']]

    three_name = '%s'%(params['ensemble'])
    three_dir = '%s'%(params['ensemble'])
    nflow = len(metadata['tau_list']) - 1

    result_3pt_chi2dof = {}
    result_3pt = {}
    result_stability = {}
    for k in R.keys():
        print('Fitting 3pt %s' % k)
        result_3pt_chi2dof[k] = {}
        result_3pt[k] = {}
        result_stability[k] = {}
        for n in range(3, 7):
            result_3pt_chi2dof[k][n] = {}
            result_3pt[k][n] = {}
            result_stability[k][n] = {}
            for tf in range(nflow + 1):
                result_3pt_chi2dof[k][n][tf] = {}
                result_3pt[k][n][tf] = {}
                result_stability[k][n][tf] = {}
                params['stability'] = {}
                for tsep in params['tsep_list']:
                    print('  n=%d, tf=%d, tsep=%d' % (n,tf,tsep))
                    middle = tsep // 2
                    prior = {'C': R[k][n][tf][tsep][0][middle]}
                    params['mtype'] = mtype = 'const'
                    params['prior'] = [prior['C']]

                    selected = {'n': {k: fit_range[k][0]}, 'tins': {k: fit_range[k][1][tsep]}}
                    fit_n = selected['n'][k]
                    # For temp speed
                    # params['tins'] = {fit_n: {k: [0, tsep // 8]}}
                    params['tins'] = {fit_n: {k: [0, 1]}}
                    params['tf'] = tf
                    params['tsep'] = tsep
                    R_onlyone = {k: np.copy(R[k][n][tf][tsep])}
                    result_chi2dof, result_good = tp.fitting_3pt(prior,params,R_onlyone,mtype,selected,correlated=True)

                    result_3pt_chi2dof[k][n][tf][tsep] = result_chi2dof
                    result_3pt[k][n][tf][tsep] = result_good
                    result_stability[k][n][tf][tsep] = params['stability'][k]

                params_plot = dict(params)
                params_plot['moment'] = n
                selected_plot = {'n': {k: fit_range[k][0]}, 'tins': {k: fit_range[k][1]}}
                data_plot = {tsep: R[k][n][tf][tsep] for tsep in params['tsep_list']}
                result_plot = {tsep: result_3pt[k][n][tf][tsep] for tsep in params['tsep_list']}
                stability_plot = {tsep: result_stability[k][n][tf][tsep] for tsep in params['tsep_list']}
                chi2_plot = {tsep: result_3pt_chi2dof[k][n][tf][tsep] for tsep in params['tsep_list']}
                tp.plot_stability_3pt(k,params_plot,selected_plot,stability_plot,chi2_plot,tit=r'$Ens=%s \quad %s$'%(params['ensemble'],inputs.labels(k)),sv='%s_%s_n%d_tf%d'%(three_name,k,n,tf),figdir='fit_three/%s'%(three_dir))
                tp.plot_result_3pt(k,data_plot,params_plot,selected_plot,result_plot,tit=r'$Ens=%s \quad %s$'%(params['ensemble'],inputs.labels(k)),sv='%s_%s_n%d_tf%d'%(three_name,k,n,tf),figdir='fit_three/%s'%(three_dir))

    return result_3pt_chi2dof, result_3pt
