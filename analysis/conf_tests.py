from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import funcs as tp
import inputs



def conf_dist(params, data2, data3, sv=''):
    tech = params['tech']
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
    Path('../%s/%s/%s/%s/'%(params['figures'],'configurations',params['ensemble'],tech)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/%s/%s/dist_%s.pdf'%(params['figures'],'configurations',params['ensemble'],tech,sv),transparent=True)
    tp.show_in_spyder()


def bin_test(params, data2, sv=''):
    tech = 'jackknife'

    Nb_list = np.arange(params['N'], 1, -1)
    mean_list = np.zeros_like(Nb_list, dtype=float)
    err_list = np.zeros_like(Nb_list, dtype=float)
    for i in tqdm.tqdm(range(len(Nb_list)), desc='Bin test'):
        Nb = Nb_list[i]
        corr_pion_0 = tp.bin_data(data2['Sigmac'], Nb).real
        params_test = {'tech': tech}

        relist = tp.resamplelist(Nb, params_test)
        relen = len(relist)
        re_corr_pion_0 = np.zeros([relen, corr_pion_0.shape[1]])
        re_corr_pion_0[0] = np.mean(corr_pion_0[relist[0]],axis=0)
        for ls in range(1, relen):
            re_corr_pion_0[ls] = np.mean(corr_pion_0[relist[ls][:-1]],axis=0)

        meff_pion_0_t = tp.cal_mass(re_corr_pion_0, 'exp', 1)[:,30]
        mean_list[i] = tp.cal_mean(meff_pion_0_t)
        err_list[i] = tp.cal_err(meff_pion_0_t, tech)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x=params['N']/Nb_list,y=mean_list,yerr=err_list,ls='None',marker='o',color='k',capsize=1,markersize=1)
    ax.set_xlabel(r'$N/Nb$')
    ax.set_xlim([0,70])
    ax.set_ylabel(r'$err(m(t=30))$')
    plt.tight_layout()
    plt.draw()
    Path('../%s/%s/%s/%s/'%(params['figures'],'configurations',params['ensemble'],tech)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/%s/%s/bin_test_%s.pdf'%(params['figures'],'configurations',params['ensemble'],tech,sv),transparent=True)


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
