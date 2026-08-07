from pathlib import Path
import os
import numpy as np
import scipy as sp
import funcs as tp
import inputs
import matplotlib.pyplot as plt
from flow_matching import c_numeric



def plotdata(params, data2, data3, mtype, two, three, three_pdf, label=''):
    tech = params['tech']
    relen = params['relen']

    fdata2 = {k:data2[k].real for k in data2.keys()}
    fdata3 = {k:data3[k].real for k in data3.keys()}

    two_name = '%s'%(params['ensemble'])
    two_tit = r'$Ens=%s$'%(params['ensemble'])
    two_dir = '%s'%(params['ensemble'])
    three_name = '%s'%(params['ensemble'])
    three_tit = r'$Ens=%s$'%(params['ensemble'])
    three_dir = '%s'%(params['ensemble'])

    t = np.arange(params['T'])

    if two:
        for k in params['key_2pt']:
            fdata2plot = {k: fdata2[k]}
            cylim = [1e-15,1e5]
            mylim = {'pion': [2.5, 4], 'kaon': [2.5, 4]}
            mtype = {'pion': 'cosh', 'kaon': 'cosh'}
            tp.plot_corr(fdata2plot,params,sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))
            tp.plot_meff(fdata2plot,params,tau=params['tau'],mtype=mtype[k],mylim=mylim[k],sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))
            tp.plot_Aeff(fdata2plot,params,tau=params['tau'],mtype=mtype[k],sv='two_%s_%s_%s'%(two_name,k,label),tit=two_tit,figdir='plot_two/%s'%(two_dir))

    if three:
        tit = three_tit
        sv = 'three_%s_%s'%(three_name,label)
        figdir = 'plot_three/%s'%(three_dir)
        ZVl = params['Z_V^l']

        for k in params['key_3pt']:
            tsep_max_3pt = params['tsep_max_3pt'][k]
            if k in ['pion', 'kaon', 'kaon_s']:
                for diag in range(1):
                    if diag == 0:
                        data_3pt = fdata3['%s-nder_0_diag_0' % k][:,0,0] * ZVl # [conf, flow, mu, tsep, tins]
                        data_2pt = fdata2['kaon' if k == 'kaon_s' else k]

                    fig, ax = plt.subplots(1,1)
                    # for tsep in range(1, tsep_max_3pt):
                    for tsep in range(1, 5):
                        x = np.arange(tsep+1) - tsep / 2
                        R = np.zeros((relen, tsep+1))
                        for ls in range(relen):
                            data_3pt_tsep = data_3pt[ls][tsep][:tsep+1]
                            # data_3pt_tsep = data_3pt[ls][tsep][:tsep+1] * (1 + np.exp(-0.156*(72-2*tsep)))
                            data_2pt_tsep = data_2pt[ls][tsep]
                            R[ls] = data_3pt_tsep / data_2pt_tsep
                        mean = tp.cal_mean(R)
                        err = tp.cal_err(R,tech=params['tech'])
                        ax.errorbar(x=x,y=mean,yerr=err,ls='-',marker='o',color=inputs.clrscm(tsep_max_3pt-1,tsep-1),mec=inputs.clrscm(tsep_max_3pt-1,tsep-1),capsize=2,fillstyle='none')
                    ax.set_xlim([-tsep_max_3pt//2 ,tsep_max_3pt//2])
                    # ax.set_ylim([0.9, 1])
                    ax.set_xlabel(r'$t_j - t_{sep}/2$')
                    ax.set_ylabel(r'$C_3(t_f;t_j;t_i) / C_2(t_f;t_i)$')
                    ax.set_title(tit)
                    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
                    plt.savefig('../%s/%s/R_%s_%s_%s.pdf'%(params['figures'],figdir,k,diag,sv),transparent=True)
                    tp.show_in_spyder()

    if three_pdf:
        tit = three_tit
        sv = 'three_%s_%s'%(three_name,label)
        figdir = 'plot_three/%s'%(three_dir)
        nflow = params['nflow']

        for k in params['key_3pt']:
            tsep_max_3pt = params['tsep_max_3pt'][k]
            if k in ['pion', 'kaon', 'kaon_s']:
                for tf in range(nflow + 1):
                    tf_GeV2 = tf * 0.125 * 1e-3 * params['hca'] ** 2
                    if tf == 0:
                        data_3pt_up = - fdata3['%s-PDF-n_3' % k][:,tf,:,:]
                        data_3pt_down = fdata3['%s-PDF-n_2' % k][:,tf,:,:] * 2.8
                    else:
                        data_3pt_up = - fdata3['%s-PDF-n_3' % k][:,tf,:,:] * c_numeric(2, tf_GeV2, 2)
                        data_3pt_down = fdata3['%s-PDF-n_2' % k][:,tf,:,:] * 2.8  * c_numeric(3, tf_GeV2, 2)

                    fig, ax = plt.subplots(1,1)
                    # for tsep in range(1, tsep_max_3pt):
                    for tsep in range(2, 5):
                        x = np.arange(tsep+1) - tsep / 2
                        R = np.zeros((relen, tsep+1))
                        for ls in range(relen):
                            data_3pt_up_tsep = data_3pt_up[ls][tsep][:tsep+1]
                            data_3pt_down_tsep = data_3pt_down[ls][tsep][:tsep+1]
                            R[ls] = data_3pt_up_tsep / data_3pt_down_tsep
                        mean = tp.cal_mean(R)
                        err = tp.cal_err(R,tech=params['tech'])
                        ax.errorbar(x=x,y=mean,yerr=err,ls='-',marker='o',color=inputs.clrscm(tsep_max_3pt-1,tsep-1),mec=inputs.clrscm(tsep_max_3pt-1,tsep-1),capsize=2,fillstyle='none')
                    ax.set_xlim([-tsep_max_3pt//2 ,tsep_max_3pt//2])
                    ax.set_ylim([-0.03, 0.03])
                    ax.set_xlabel(r'$t_j - t_{sep}/2$')
                    ax.set_ylabel(r'$\langle x^2 \rangle / \langle x \rangle$')
                    ax.set_title(tit)
                    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
                    plt.savefig('../%s/%s/x3_x2_tf_%d_%s_%s_%s.pdf'%(params['figures'],figdir,tf,k,diag,sv),transparent=True)
                    tp.show_in_spyder()
