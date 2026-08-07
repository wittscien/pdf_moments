from pathlib import Path
import os
import numpy as np
import scipy as sp
import funcs as tp
import inputs
import matplotlib.pyplot as plt
from flow_matching import c_numeric



def plotdata(params, data2, data3, metadata, two, three, three_pdf, result, label=''):
    tech = params['tech']
    relen = params['relen']

    fdata2 = {k:data2[k].real for k in data2.keys()}
    fdata3 = {k:{dt:data3[k][dt].real for dt in data3[k].keys()} for k in data3.keys()}

    two_name = '%s'%(params['ensemble'])
    two_tit = r'$Ens=%s$'%(params['ensemble'])
    two_dir = '%s'%(params['ensemble'])
    three_name = '%s'%(params['ensemble'])
    three_tit = r'$Ens=%s$'%(params['ensemble'])
    three_dir = '%s'%(params['ensemble'])

    T = params['T']
    t = np.arange(T)

    if two:
        for k in params['key_2pt']:
            fdata2plot = {k: fdata2[k]}
            cylim = [1e-15,1e5]
            mylim = {'pion': [0, 0.5], 'kaon': [0, 0.5]}
            mtype = {'pion': 'cosh', 'kaon': 'cosh'}
            # tp.plot_corr(fdata2plot,params,sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))
            tp.plot_meff(fdata2plot,params,tau=params['tau'],mtype=mtype[k],mylim=mylim[k],sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))
            # tp.plot_Aeff(fdata2plot,params,tau=params['tau'],mtype=mtype[k],sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))

    if three:
        sv = 'three_%s_%s'%(three_name,label)
        figdir = 'plot_three/%s'%(three_dir)
        ZVl = params['Z_V^l']
        ZVs = params['Z_V^s']
        len_tsep_list = len(params['tsep_list'])
        result_use = {'pion': [[0, 0.08559]], 'kaon': [[0, 0.17888]]} if result is None else result

        for k in params['key_3pt']:
            if k == 'pion':
                k_one = 'pion'
            elif k in ['kaon', 'kaon_s']:
                k_one = 'kaon'
            tit = r'$Ens=%s \quad %s$' % (params['ensemble'], k)
            data_2pt = fdata2['kaon' if k == 'kaon_s' else k] # [ls, tsep]

            fig, ax = plt.subplots(1,1)
            for itsep, tsep in enumerate(params['tsep_list']):
                data_3pt = fdata3['%s-PDF-n_1' % k][tsep][:,0,:] # [tsep][ls, flow, tins]
                x = np.arange(tsep+1) - tsep / 2
                R = np.zeros((relen, tsep+1))
                for ls in range(relen):
                    m_one = result_use[k_one][ls][1]
                    if k == 'pion':
                        data_3pt_tsep = data_3pt[ls] * ZVl * (1 + np.exp(-m_one * (T-2*tsep)))
                    elif k == 'kaon':
                        data_3pt_tsep = data_3pt[ls] * ZVs * (1 + np.exp(-m_one*(T-2*tsep)))
                    elif k == 'kaon_s':
                        data_3pt_tsep = - data_3pt[ls] * ZVs * (1 + np.exp(-m_one*(T-2*tsep)))
                    data_2pt_tsep = data_2pt[ls][tsep]
                    R[ls] = data_3pt_tsep / data_2pt_tsep
                mean = tp.cal_mean(R)
                err = tp.cal_err(R,tech=params['tech'])
                ax.errorbar(x=x,y=mean,yerr=err,ls='-',marker='o',color=inputs.clrscm(len_tsep_list,itsep),mec=inputs.clrscm(len_tsep_list,itsep),capsize=2,fillstyle='none')
            ax.set_xlim([-max(params['tsep_list'])//2 ,max(params['tsep_list'])//2])
            ax.set_ylim([0.5, 1.5])
            ax.set_xlabel(r'$t_j - t_{sep}/2$')
            ax.set_ylabel(r'$C_3(t_f;t_j;t_i) / C_2(t_f;t_i)$')
            ax.set_title(tit)
            Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
            plt.savefig('../%s/%s/R_%s_%s.pdf'%(params['figures'],figdir,k,sv),transparent=True)
            tp.show_in_spyder()

    if three_pdf:
        tit = three_tit
        sv = 'three_%s_%s'%(three_name,label)
        figdir = 'plot_three/%s'%(three_dir)
        len_tsep_list = len(params['tsep_list'])
        nflow = len(metadata['tau_list']) - 1

        for k in params['key_3pt']:
            if k == 'pion':
                k_one = 'pion'
            elif k in ['kaon', 'kaon_s']:
                k_one = 'kaon'
            tit = r'$Ens=%s \quad %s$' % (params['ensemble'], k)
            # n is the number of mu
            for n in range(3, 7):
                for tf in range(nflow + 1):
                    tf_GeV2 = tf * metadata['flow_dt'] * params['hca'] ** 2

                    fig, ax = plt.subplots(1,1)
                    for itsep, tsep in enumerate(params['tsep_list']):
                        x = np.arange(tsep+1) - tsep / 2
                        R = np.zeros((relen, tsep+1))
                        for ls in range(relen):
                            m_one = result_use[k_one][ls][1]

                            data_3pt_up = fdata3['%s-PDF-n_%d' % (k, n)][tsep][:,tf,:]
                            data_3pt_down = fdata3['%s-PDF-n_2' % (k)][tsep][:,tf,:] * (- m_one) ** (n - 1)
                            if tf != 0:
                                data_3pt_up *= c_numeric(2, tf_GeV2, 2)
                                data_3pt_down *= c_numeric(n + 1, tf_GeV2, 2)

                            data_3pt_up_tsep = data_3pt_up[ls]
                            data_3pt_down_tsep = data_3pt_down[ls]
                            R[ls] = data_3pt_up_tsep / data_3pt_down_tsep
                        mean = tp.cal_mean(R)
                        err = tp.cal_err(R,tech=params['tech'])
                        ax.errorbar(x=x,y=mean,yerr=err,ls='-',marker='o',color=inputs.clrscm(len_tsep_list,itsep),mec=inputs.clrscm(len_tsep_list,itsep),capsize=2,fillstyle='none')
                    ax.set_xlim([-max(params['tsep_list'])//2 ,max(params['tsep_list'])//2])
                    # ax.set_ylim([-0.03, 0.03])
                    ax.set_xlabel(r'$t_j - t_{sep}/2$')
                    ax.set_ylabel(r'$\langle x^%d \rangle / \langle x \rangle$' % (n - 1))
                    ax.set_title(tit)
                    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
                    plt.savefig('../%s/%s/R_x%d_x2_tf_%d_%s_%s.pdf'%(params['figures'],figdir,n,tf,k,sv),transparent=True)
                    tp.show_in_spyder()
