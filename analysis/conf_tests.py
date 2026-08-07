from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import funcs as tp
import inputs



def conf_dist(params, data2, data3, sv=''):
    fig, ax = plt.subplots(1, 1)

    tsep = 20

    corr_value = data2['pion'][:,tsep]
    ax.scatter(x=np.arange(len(params['confs'])),y=corr_value,marker='o',color='k', label='2pt')

    # ZVl = params['Z_V^l']
    # corr_value = data3['pion-nder_0_diag_0'][:,0,0,tsep,tsep//2] * ZVl
    # ax.scatter(x=params['confs'],y=corr_value,marker='o',color='r', label='3pt')

    corr_value = data3['pion-PDF-n_1'][32][:,1,16]
    ax.scatter(x=np.arange(len(params['confs'])),y=corr_value,marker='o',color='r', label='3pt')

    ax.set_xlabel('conf #')
    ax.set_ylabel(r'$C(tsep = %d)$'%(tsep))
    plt.tight_layout()
    ax.legend()
    Path('../%s/%s/%s/'%(params['figures'],'configurations',params['ensemble'])).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/%s/dist_%s.pdf'%(params['figures'],'configurations',params['ensemble'],sv),transparent=True)
    tp.show_in_spyder()


def bin_test(params, data2, sv=''):
    tech = 'jackknife'
    t = params['T'] // 2

    Nb_list = np.arange(params['N'], 1, -1)
    mean_list = np.zeros_like(Nb_list, dtype=float)
    err_list = np.zeros_like(Nb_list, dtype=float)
    for i in tqdm.tqdm(range(len(Nb_list)), desc='Bin test'):
        Nb = Nb_list[i]
        corr_pion = tp.bin_data(data2['pion'], Nb).real
        params_test = {'tech': tech}

        relist = tp.resamplelist(Nb, params_test)
        re_corr_pion = tp.resample_general(corr_pion, tech, relist)
        re_corr_pion = (np.roll(re_corr_pion[:,::-1], 1, axis=1) + re_corr_pion) / 2

        meff_pion_t = np.array([tp.cal_mass(corr, 'cosh', 1)[t] for corr in re_corr_pion])
        mean_list[i] = tp.cal_mean(meff_pion_t)
        err_list[i] = tp.cal_err(meff_pion_t, tech)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x=params['N']/Nb_list,y=mean_list,yerr=err_list,ls='None',marker='o',color='k',capsize=1,markersize=1)
    ax.set_xlabel(r'$N/Nb$')
    ax.set_xlim([0,71])
    ax.set_ylabel(r'$m_{\mathrm{eff}}(t=%d)$'%t)
    plt.tight_layout()
    plt.draw()
    Path('../%s/%s/%s/'%(params['figures'],'configurations',params['ensemble'])).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/%s/bin_test_%s.pdf'%(params['figures'],'configurations',params['ensemble'],sv),transparent=True)
    tp.show_in_spyder()


def boots_test(params, data2, sv=''):
    tech = 'bootstrap'

    Nb = len(data2['Sigmac'])
    Nb = 50
    corr_pion_0 = tp.bin_data(data2['Sigmac'], Nb).real
    Nboots_list = np.arange(2, 400, 1)
    err_list = np.zeros_like(Nboots_list, dtype=float)
    for i in tqdm.tqdm(range(len(Nboots_list)), desc='Boots test'):
        Nbs = Nboots_list[i]
        relen = Nbs + 1
        params_test = {'tech': tech, 'seed': 0, 'Nbs': Nbs, 'Mbs': Nb}
        relist = tp.resamplelist(Nb, params_test)
        re_corr_pion_0 = np.zeros([relen, corr_pion_0.shape[1]])
        for ls in range(relen):
            re_corr_pion_0[ls] = np.mean(corr_pion_0[relist[ls]],axis=0)
        meff_pion_0_t = tp.cal_mass(re_corr_pion_0.real, 'exp', 1)[:,13]
        err_list[i] = tp.cal_err(meff_pion_0_t, tech)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x=Nboots_list,y=err_list,marker='o',color='k',s=1)
    ax.set_xlabel('Nboots')
    ax.set_ylabel(r'$err(m(t=13))$')
    plt.tight_layout()
    plt.draw()
    Path('../%s/%s/%s/%s/'%(params['figures'],'configurations',params['ensemble'],tech)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/%s/%s/boots_test_%s.pdf'%(params['figures'],'configurations',params['ensemble'],tech,sv),transparent=True)
