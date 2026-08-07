from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs


def fit_two(params, data2, mtype, obj):
    # Parameter structure: A0, E0, A1, E1, ...
    tau = params['tau']
    tech = params['tech']
    fdata2 = {k:data2[k].real for k in data2.keys()}

    one_name = '%s'%(params['ensname'])
    one_tit = r'$Ens=%s$'%(params['ensname'])
    one_dir = '%s/%s'%(params['ensname'],tech)

    # Write priors to txt
    Path('priors/%s/'%(one_dir)).mkdir(parents=True, exist_ok=True)
    prior_txt = open("priors/%s/prior_%s.txt"%(one_dir,one_name), 'w')
    prior_txt.close()

    T = params['T']

    result_para = {}
    result_chi2dof = {}
    result = {}
    ans = {}
    for k in fdata2.keys():
        # if k.split('_')[0] == 'eta': continue
        #if k != 'pion_0': continue
        #if k.split('_')[0] != 'pion': continue
        print(k)
        par = k.split('_')[0]
        mom = int(k.split('_')[1])

        # Lazy priors
        prior = {}
        lazy_time = T // 2
        prior['E_0'] = tp.cal_mass(fdata2[k][0],mtype=mtype)[lazy_time]
        prior['E_1'] = prior['E_0'] + 0.4
        prior['A_0'] = tp.cal_A(fdata2[k][0],mtype=mtype)[lazy_time]
        prior['A_1'] = prior['A_0'] / 10

        params['mtype'] = mtype
        params['prior'] = [prior['A_0'], prior['E_0'], prior['A_1'], prior['E_1']]

        if obj == 'corr':
            # corr fit
            fdata2_onlyone = {}
            fdata2_onlyone[k] = np.copy(fdata2[k])
            #[selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fdata2_onlyone,mtype,selected,correlated=False)
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fdata2_onlyone,mtype,selected)
        elif obj == 'meff':
            # meff fit
            fmdata2_onlyone = {}
            fmdata2_onlyone[k] = tp.cal_mass(fdata2[k],mtype,tau)
            params['mtype'] = 'const'
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fmdata2_onlyone,'const',selected)

        print('Single fit result for %s of Ensemble %s:'%(k,params['ensname']))
        if obj == 'corr':
            print('m = '+repr(gv.gvar(ans[k]['mean'][1],ans[k]['err'][1]))+'\tA0 = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][1]*params['hca'],ans[k]['err'][1]*params['hca']))+' MeV')
            if selected['n'] == 2:
                print('E1 = '+repr(gv.gvar(ans[k]['mean'][3],ans[k]['err'][3]))+'\tA1 = '+repr(gv.gvar(ans[k]['mean'][2],ans[k]['err'][2])))

                prior_txt = open("priors/%s/prior_%s.txt"%(one_dir,one_name), 'a')
                prior_txt.write("""            if (par == '%s'):
                        if mom == %d:
                            prior['A_0'] = %.5f
                            prior['A_1'] = %.5f
                            prior['E_0'] = %.5f
                            prior['E_1'] = %.5f\n"""%(par,mom,ans[k]['mean'][0],ans[k]['mean'][2],ans[k]['mean'][1],ans[k]['mean'][3]))
                prior_txt.close()
        elif obj == 'meff':
            print('m = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][0]*params['hca'],ans[k]['err'][0]*params['hca']))+' MeV')

        if params['lazy_tmin']:
            selected['tmin'][k] = selected_lazy_tmin

        # Plot stability
        tp.plot_stability(k,1,params,selected,result_para[k],result_chi2dof[k],tit=r'%s %s'%(one_tit,inputs.labels(k)),sv='%s_%s'%(one_name,k),figdir='fit_one/%s'%(one_dir))

        # Plot results
        fdata2_plot = {}
        fdata2_plot[k] = np.copy(fdata2[k])
        tp.plot_result(fdata2_plot,params,selected,result[k],ans[k],tau=tau,mtype=params['mtype'],tit=r'%s %s'%(one_tit,inputs.labels(k)),sv='%s_%s'%(one_name,k),figdir='fit_one/%s'%(one_dir))

    return fdata2, result_para, result_chi2dof, result, ans
