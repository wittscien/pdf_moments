from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs
import fitting_ranges


def fit_two(params, data2, mtype, obj):
    # Parameter structure: A0, E0, A1, E1, ...
    tau = params['tau']
    T = params['T']
    fdata2 = {k:data2[k].real for k in data2.keys()}

    fit_range = fitting_ranges.ranges_two[params['ensemble']]
    selected = {'n': {}, 'tmin': {}, 'tmax': {}}
    params['tmax'] = {1:{}, 2:{}}
    params['tmin'] = {1:{}, 2:{}}
    for k in fdata2.keys():
        selected['n'][k], selected['tmin'][k], selected['tmax'][k] = fit_range[k]
        params['tmax'][1][k] = [selected['tmax'][k], selected['tmax'][k]]
        params['tmax'][2][k] = [selected['tmax'][k], selected['tmax'][k]]
        params['tmin'][1][k] = [2, selected['tmax'][k] - 1]
        params['tmin'][2][k] = [2, selected['tmax'][k] - 3]

    two_name = '%s'%(params['ensemble'])
    two_tit = r'$\mathrm{Ens}=%s$'%(params['ensemble'])
    two_dir = '%s'%(params['ensemble'])

    result_para = {}
    result_chi2dof = {}
    result = {}
    ans = {}
    for k in fdata2.keys():
        print(k)

        # Lazy priors
        prior = {}
        lazy_time = T // 2
        prior['E_0'] = tp.cal_mass(fdata2[k][0],mtype=mtype,tau=tau)[lazy_time]
        prior['E_1'] = prior['E_0'] + 0.4
        prior['A_0'] = tp.cal_A(fdata2[k][0],mtype=mtype,tau=tau)[lazy_time]
        prior['A_1'] = prior['A_0'] / 10

        params['mtype'] = mtype
        params['prior'] = [prior['A_0'], prior['E_0'], prior['A_1'], prior['E_1']]

        if obj == 'corr':
            # corr fit
            fdata2_onlyone = {}
            fdata2_onlyone[k] = np.copy(fdata2[k])
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fdata2_onlyone,mtype,selected,correlated=True)
        elif obj == 'meff':
            # meff fit
            fmdata2_onlyone = {}
            fmdata2_onlyone[k] = tp.cal_mass(fdata2[k],mtype,tau)
            params['mtype'] = 'const'
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fmdata2_onlyone,'const',selected,correlated=True)

        print('Single fit result for %s of Ensemble %s:'%(k,params['ensemble']))
        if obj == 'corr':
            print('m = '+repr(gv.gvar(ans[k]['mean'][1],ans[k]['err'][1]))+'\tA0 = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][1]*params['hca'],ans[k]['err'][1]*params['hca']))+' MeV')
            if selected['n'][k] == 2:
                print('E1 = '+repr(gv.gvar(ans[k]['mean'][3],ans[k]['err'][3]))+'\tA1 = '+repr(gv.gvar(ans[k]['mean'][2],ans[k]['err'][2])))
        elif obj == 'meff':
            print('m = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][0]*params['hca'],ans[k]['err'][0]*params['hca']))+' MeV')

        if params['lazy_tmin']:
            selected['tmin'][k] = selected_lazy_tmin

        # Plot stability
        tp.plot_stability(k,1,params,selected,result_para[k],result_chi2dof[k],tit=r'%s\quad $%s$'%(two_tit,inputs.labels(k)),sv='%s_%s'%(two_name,k),figdir='fit_two/%s'%(two_dir))

        # Plot results
        fdata2_plot = {}
        fdata2_plot[k] = np.copy(fdata2[k])
        tp.plot_result(fdata2_plot,params,selected,result[k],ans[k],tau=tau,mtype=params['mtype'],tit=r'%s\quad $%s$'%(two_tit,inputs.labels(k)),sv='%s_%s'%(two_name,k),figdir='fit_two/%s'%(two_dir))

    return fdata2, result_para, result_chi2dof, result, ans
