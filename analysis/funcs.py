import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
#import warnings
import struct
from scipy.optimize import least_squares, OptimizeResult
from scipy.optimize import fsolve
from scipy import linalg
import pickle
import inputs
#from itertools import combinations_with_replacement
import tqdm


def show_in_spyder():
    # Spyder sets this environment variable. In VS Code or terminal runs, save
    # the figure but do not pop up a window.
    if 'SPYDER_ARGS' in os.environ:
        plt.show()



def gen_dist_list(L):
    dist_list = np.zeros((L // 2) ** 2 * 3 + 1)
    for x in range(-L // 2, L // 2):
        for y in range(-L // 2, L // 2):
            for z in range(-L // 2, L // 2):
                dist = x ** 2 + y ** 2 + z ** 2
                dist_list[dist] += 1
    # If multiplied by L ** 3, then it is the factor appeared in the contraction code that sums over all possible coordinates of two currents. But there we have two volume loops, here we just need the 1,6,12... list.
    dist_list *= L ** 3
    return dist_list


def write_data(file,data):
    f = open(file,'wb')
    pickle.dump(data,f)
    f.close()


def fast_read(file):
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def jackknife_relist(thing):
    # Use already resampled matrix and its \theta^hat
    N = thing.shape[0] - 1
    mean = thing[0]
    sigma = np.sqrt(np.sum((thing[1:] - mean) ** 2, axis=0) * (N - 1) / N)
    return sigma


def bootstrap_relist(thing):
    # Use already resampled matrix and its \theta^hat
    tilde = np.mean(thing[1:], axis=0)
    sigma = np.sqrt(np.mean((thing[1:] - tilde) ** 2, axis=0))
    return sigma


def bootstrap_cov(thing):
    N = thing.shape[0] - 1
    mean = thing[0]
    X = thing - mean
    N = thing.shape[0] - 1
    C = np.einsum('ni, nj -> ij', X, X) / N
    return C


def jackknife_cov(thing):
    N = thing.shape[0] - 1
    mean = thing[0]
    X = thing - mean
    N = thing.shape[0] - 1
    C = np.einsum('ni, nj -> ij', X, X) / N
    C *= (N - 1)
    return C


def resamplelist(length,tparams):
    if (tparams['tech'] == 'bootstrap'):
        np.random.seed(tparams['seed'])
        return np.concatenate((np.reshape(np.arange(length), (1,-1)), np.random.randint(0,length,(tparams['Nbs'],tparams['Mbs']))))
    elif (tparams['tech'] == 'jackknife'):
        jk_lst = np.zeros([length+1,length],dtype=int)
        jk_lst[0] = np.arange(length)
        for i in range(length):
            jk_lst[i+1] = np.concatenate((np.arange(0,i),np.arange(i+1,length),[10000]))
        return jk_lst


def cal_mean(thing):
    return thing[0]


def cal_err(thing,tech):
    if (tech == 'bootstrap'):
        return bootstrap_relist(thing)
    elif (tech == 'jackknife'):
        return jackknife_relist(thing)


def cal_cov(thing,tech):
    # Cannot use np.cov because we have to use thing[0]
    if (tech == 'bootstrap'):
        return bootstrap_cov(thing)
    elif (tech == 'jackknife'):
        return jackknife_cov(thing)


def cov_ellipse(cov):
    [a, b, c] = [cov[0,0], cov[0,1], cov[1,1]]
    lambda1 = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    lambda2 = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    if b == 0  and a >= c:
        theta = 0
    elif b == 0 and a < c:
        theta = np.pi / 2
    else:
        theta = math.atan2(lambda1 - a, b)
    return lambda1, lambda2, theta


def bin_data(data, Nb):
    N = len(data)
    target_bin_size = N / Nb
    binned_data = []
    start = 0.0

    for i in range(Nb):
        end = start + target_bin_size
        start_int = int(np.floor(start))
        end_int = int(np.floor(end))

        fraction_start = 1 - (start - start_int)
        fraction_end = end - end_int
        # First block
        bin_slice = data[start_int] * fraction_start
        # Ending block, avoid exceeding N
        if not np.isclose(fraction_end, 0): bin_slice += data[end_int] * fraction_end
        # Middle blocks
        if end_int > start_int + 1:
            bin_slice += np.sum(data[start_int + 1:end_int], axis=0)

        binned_data.append(bin_slice / target_bin_size)
        start = end

    return np.array(binned_data)


def resample(data, tech, relist):
    relen = relist.shape[0]
    T = data.shape[1]
    redata = np.zeros([relen, T], dtype = complex)
    if tech == 'bootstrap':
        for ls in range(relen):
            redata[ls] = np.mean(data[relist[ls]],axis=0)
    elif tech == 'jackknife':
        redata[0] = np.mean(data[relist[0]],axis=0)
        for ls in range(1, relen):
            redata[ls] = np.mean(data[relist[ls][:-1]],axis=0)
    return redata


def resample_general(data, tech, relist):
    relen = relist.shape[0]
    redata = np.zeros((relen,) + data.shape[1:], dtype = complex)
    if tech == 'bootstrap':
        for ls in range(relen):
            redata[ls] = np.mean(data[relist[ls]],axis=0)
    elif tech == 'jackknife':
        redata[0] = np.mean(data[relist[0]],axis=0)
        for ls in range(1, relen):
            redata[ls] = np.mean(data[relist[ls][:-1]],axis=0)
    return redata


def resample4_parallel(data_ls, tech, relist):
    return np.mean(data_ls,axis=0)

def resample4(data, tech, relist):
    relen = relist.shape[0]
    tJJ_max = data.shape[1]
    len_dist = data.shape[2]
    redata = np.zeros([relen, tJJ_max, len_dist, 4], dtype = complex)
    if tech == 'bootstrap':
        for ls in range(relen):
            redata[ls] = np.mean(data[relist[ls]],axis=0)
    elif tech == 'jackknife':
        redata[0] = np.mean(data[relist[0]],axis=0)
        for ls in range(1, relen):
            redata[ls] = np.mean(data[relist[ls][:-1]],axis=0)
        # Not tested
        # with Pool(processes = os.cpu_count()) as pool:
        #     pool_result = pool.starmap(resample4_parallel, [(data[relist[ls][:-1]],tech,relist) for ls in range(1, relen)])
        # for ls in range(len(pool_result)):
        #     redata[ls] = pool_result[ls]
    return redata


def plot_corr(data, params, ylog=True, cylim='no', sv='', tit='', figdir=''):
    # Use full data
    fig, ax = plt.subplots(1,1)
    data_mean = {}
    data_err = {}
    ylimup = 0
    ylimdown = 1e100
    for i,k in enumerate(data.keys()):
        T = data[k].shape[1]
        data_mean[k] = cal_mean(data[k])
        data_err[k] = cal_err(data[k],params['tech'])
        x = np.arange(data[k].shape[1])
        ax.errorbar(x=x,y=data_mean[k],yerr=data_err[k],ls='None',marker='o',color=inputs.clrscm(len(data),i),mec=inputs.clrscm(len(data),i),capsize=2,fillstyle='none',label=inputs.labels(k))
        ylimup = max(ylimup,np.max(data_mean[k][1:T//2-1]) * 100)
        ylimdown = min(ylimdown,np.min(data_mean[k][1:T//2-1]) / 100)
    #ylimdown=-1e30
    #ylimup=1e30
    ax.axis([-0.2,params['T']//2,ylimdown,ylimup])
    if cylim != 'no':
        ax.set_ylim(cylim)
    ax.legend()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('correlator')
    if ylog == True:
        ax.set_yscale('symlog',linthresh=1e-35)
        #ax.set_yscale('log')
    ax.set_title(tit)
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/corr_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    show_in_spyder()


def cal_mass(data,mtype='exp',tau=1):
    # Use mean data
    data = data.real
    T = data.shape[0]
    def func_cosh(mef,i):
        return data[i] / np.roll(data,-1)[i] - np.cosh(mef*(i-T/2)) / np.cosh(mef*(i+tau-T/2))
    
    #def func_S3(mef,i):
    #    return data[i] / np.roll(data,-1)[i] - (np.cosh(mef*(params['T']/2-i))/np.cosh(params['m']*(params['T']/2-i))-np.cosh(mef*(params['T']/2-i-1))/np.cosh(params['m']*(params['T']/2-i-1))) / (np.cosh(mef*(params['T']/2-i-tau))/np.cosh(params['m']*(params['T']/2-i-tau))-np.cosh(mef*(params['T']/2-i-1-tau))/np.cosh(params['m']*(params['T']/2-i-1-tau)))
    if (mtype == 'exp') or (mtype == 'Gexp'):
        meff = 1./tau * np.log(data / np.roll(data,-tau))
    elif (mtype == 'cosh') or (mtype == 'sinh') or (mtype == 'Gcosh') or (mtype == 'Gsinh'):
        meff = 1./tau * np.arccosh((np.roll(data,-tau)+np.roll(data,tau))/2/data)
        #meff = np.zeros(data.shape[0])
        #for i in range(data.shape[0]):
        #    meff[i] = fsolve(func_cosh,3,args=(i))
    elif (mtype == 'const'): # treat as cosh
        meff = 1./tau * np.arccosh((np.roll(data,-tau)+np.roll(data,tau))/2/data)
    #elif mtype == 'S3':
    #    meff = np.zeros(data.shape[0])
    #    for i in range(data.shape[0]):
    #        meff[i] = fsolve(func_S3,3,args=(i))
    return meff


def cal_A(data,mtype='exp',tau=1):
    # Effective ground-state amplitude calculated with the effective mass.
    T = data.shape[0]
    x = np.arange(T)
    meff = cal_mass(data,mtype=mtype,tau=tau)
    if mtype == 'exp':
        Aeff = np.exp(meff * x) * data
    elif mtype == 'cosh':
        Aeff = np.exp(meff * T / 2) / (2 * np.cosh(meff * (x - T / 2))) * data
    elif mtype == 'sinh':
        Aeff = np.exp(meff * T / 2) / (4 * np.sinh(meff/2) * np.sinh(meff * (x - T / 2 + 1/2))) * data
    return Aeff


def plot_meff(data,params_test,tau=1,mtype='exp',mylim='no',sv='',tit='',figdir='',m_pi='no'):
    # 2025.3.14: Happy Pi festival! The weight will be taken care in the correlation matrix.
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6.4, 4)
    meff = {}
    meff_mean = {}
    meff_err = {}
    params=dict(params_test)
    #params_original = dict(params)
    #del params['m']
    params.pop('m',None)
    ylimup = -100
    ylimdown = 100
    for i,k in enumerate(data.keys()):
        T = data[k].shape[1]
        x = np.arange(data[k].shape[1])
        meff[k] = np.zeros_like(data[k])
        for j in range(data[k].shape[0]):
            meff[k][j] = cal_mass(data[k][j],mtype=mtype,tau=tau)
        meff_mean[k] = cal_mean(meff[k])
        meff_err[k] = cal_err(meff[k],tech=params['tech'])
        mask_large = ((meff_err[k] / np.abs(meff_mean[k])) > 0.3) | np.isnan(meff_mean[k]) | np.isnan(meff_err[k])
        mask_normal = ~mask_large
        ax.errorbar(x=x[mask_normal]+0.05*i,y=meff_mean[k][mask_normal],yerr=meff_err[k][mask_normal],ls='None',marker='o',color=inputs.clrscm(len(data),i),mec=inputs.clrscm(len(data),i),capsize=2,fillstyle='none',label=inputs.labels(k))
        ax.errorbar(x=x[mask_large]+0.05*i,y=meff_mean[k][mask_large],yerr=meff_err[k][mask_large],ls='None',marker='o',color=inputs.clrscm(len(data),i),mec=inputs.clrscm(len(data),i),capsize=2,fillstyle='none',alpha=0.1)
        if (not all(np.isnan(i) for i in meff_mean[k])):
            #ylimup = max(ylimup,(np.nanmax(meff_mean[k][3:T//2-3]) - np.nanmin(meff_mean[k][3:T//2-3])) / 2 + np.nanmin(meff_mean[k][3:T//2-3]))
            #ylimdown = min(ylimdown,np.nanmin(meff_mean[k][3:T//2-3]) - meff_err[k][np.nanargmin(meff_mean[k][3:T//2-3])+3] * 10)
            ylimup = max(ylimup,np.nanmax(meff_mean[k][int(0.15*T):T//2-3]))
            ylimdown = min(ylimdown,np.nanmin(meff_mean[k][int(0.15*T):T//2-3]))
    dif = (ylimup - ylimdown) / 5
    ax.axis([-0.2,T//2+3,ylimdown-dif,ylimup])
    ax.axis([-0.2,T//2,ylimdown-dif,ylimup])
    ax.set_xlim([0,30])
    #ax.axis([-0.2,T,0,3])
    if mylim != 'no':
        ax.set_ylim(mylim)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$m_{\mathrm{eff}}$')
    # In mpi unit
    if m_pi != 'no':
        aE_to_Edmpi = lambda aE: aE / m_pi
        Edmpi_to_aE = lambda Edmpi: Edmpi * m_pi
        secax_y = ax.secondary_yaxis('right', functions=(aE_to_Edmpi, Edmpi_to_aE))
        secax_y.set_ylabel(r'$m_{\mathrm{eff}} / M_{\pi}$')
    ax.set_title(tit)
    # ax.legend()
    # plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/meff_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    show_in_spyder()


def plot_Z(params,E0_matrix,corrmat,vecs,t0,tv,tech,colors='b',figdir='',sv=''):
    # Using Gresult_mean, [0]
    # v_ti has been sorted and has a shape of [relen, T, n, n]
    # Dudek rho pipi PRD 2013: Eq. 3
    relen = corrmat.shape[0]
    N = corrmat.shape[1]
    T = corrmat.shape[3]
    Z_ni = np.zeros((relen, T, N, N))
    for t in range(T):
        for ls in range(relen):
            vdagger_C_ls = contract("nj, ji -> ni", vecs[ls][t].conj().T, corrmat[ls,:,:,t0])
            for n in range(N):
                E0 = E0_matrix[ls][n]
                Z_ni[ls][t][n] = np.abs(np.sqrt(2 * E0) * np.exp(E0 * t0 / 2) * vdagger_C_ls[n,:])
            # normalization with mean, the maxim in each column
            if ls == 0 and t == tv: norm = np.max(Z_ni[ls][t], axis=0)
    for t in range(T):
        for ls in range(relen):
            for i in range(N):
                Z_ni[ls][t][:,i] /= norm[i]

    # 1. The plateau of Z
    fig, ax = plt.subplots(1, N, figsize=(N*6,4), sharey=True)
    Z_ni_mean = cal_mean(Z_ni)
    Z_ni_err = cal_err(Z_ni,tech=tech)
    x = np.arange(T)
    for n in range(N):
        for i in range(N):
            ax[n].errorbar(x=x+0.05*i,y=Z_ni_mean[:,n,i],yerr=Z_ni_err[:,n,i],ls='None',marker='o',color=inputs.clrscm(N,i),mec=inputs.clrscm(N,i),capsize=2,fillstyle='none',label='op %d'%(i))
        ax[n].set_title(r'$E_{%d}$' % n)
        ax[n].axis([-0.2,T//2+3,0,1])
        ax[n].set_xlabel(r'$t$')
        ax[n].legend()
    ax[0].set_ylabel(r'$Z$')
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/GEVPZ_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    plt.show()


    # 2. The histogram of Z at tv
    fig, ax = plt.subplots(1, N, figsize=(N*3,4), sharey=True)
    width = 0.2
    for n in range(N):
        yerr = Z_ni_err[tv,n] if not np.any(np.isnan(Z_ni_err[tv,n])) else 1
        ax[n].bar(np.arange(N)*width, Z_ni_mean[tv,n], yerr=yerr, color=colors, width=width, alpha=0.9, capsize=2)
        ax[n].set_title(r'$E_{%d}$' % n)
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        ax[n].set_ylim(0, 1)
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/overlap_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    plt.show()


def plot_Aeff(data,params,tau=1,mtype='exp',sv='',tit='',figdir=''):
    T = params['T']
    Aeff = {}
    Aeff_mean = {}
    Aeff_err = {}
    fig, axes = plt.subplots(len(data),1,figsize=(6,len(data)*0.5),gridspec_kw=dict(hspace=0),sharex=True)
    for i,k in enumerate(data.keys()):
        x = np.arange(data[k].shape[1])
        Aeff[k] = np.zeros_like(data[k])
        for j in range(data[k].shape[0]):
            Aeff[k][j] = cal_A(data[k][j],mtype=mtype,tau=tau)
        Aeff_mean[k] = cal_mean(Aeff[k])
        Aeff_err[k] = cal_err(Aeff[k],tech=params['tech'])
        if len(data) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.errorbar(x=x,y=Aeff_mean[k],yerr=Aeff_err[k],ls='None',marker='o',color=inputs.clrs[i],mec=inputs.clrs[i],capsize=2,fillstyle='none',label=inputs.labels(k))
        ylimup = (np.nanmax(Aeff_mean[k][int(0.15*T):T//2-6]) - np.nanmin(Aeff_mean[k][int(0.15*T):T//2-6])) + np.nanmin(Aeff_mean[k][int(0.15*T):T//2-6])
        #ylimdown = np.nanmin(Aeff_mean[k][int(0.15*T):T//2]) - Aeff_err[k][np.nanargmin(Aeff_mean[k][int(0.15*T):T//2])+3] * 2
        ylimdown = -(np.nanmax(Aeff_mean[k][int(0.15*T):T//2-6]) - np.nanmin(Aeff_mean[k][int(0.15*T):T//2-6])) + np.nanmin(Aeff_mean[k][int(0.15*T):T//2-6])
        lim = [0,T//2+3,ylimdown,ylimup]
        #lim = [0,len(Aeff_mean[k])/2,0,30]
        #lim = [xlim[0],xlim[1],ylim[k][0],ylim[k][1]]
        if np.isfinite(ylimdown) and np.isfinite(ylimup):
            ax.axis(lim)
        if len(data) == 1 or ((len(data) != 1) and i == len(data) - 1):
            ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$A_0$')
        ax.legend(loc='upper left')
    plt.title(tit)
    plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/Aeff_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    show_in_spyder()


def plot_stability(k,fit_num,paramso,selectedo,result,result_chi,tit='',sv='',figdir=''):
    # Should use bootstraped result
    tau = paramso['tau']
    fig, axes = plt.subplots(2,1,sharex=True,gridspec_kw=dict(height_ratios=[4,1],hspace=0))
    fig.set_size_inches(6.4, 4)
    params = dict(paramso)
    selected = dict(selectedo)
    selected['tmin'] = selectedo['tmin'][k]
    selected['tmax'] = selectedo['tmax'][k]
    selected['n'] = selectedo['n'][k]
    ylimup = 0
    ylimdown = 0
    tmin00 = 1000
    tmin11 = -1
    for n in range(params['ns_min'],params['ns_max']+1):
        if params['mtype'] == 'const':
            if n != 1: continue
        tmin0 = params['tmin'][n][k][0]
        tmin1 = params['tmin'][n][k][1]
        tmin00 = min(tmin00, tmin0)
        tmin11 = max(tmin11, tmin1)
        x = np.arange(tmin0,tmin1+1,tau)
        E0_mean = np.zeros_like(x,dtype=float)
        E0_err = np.zeros_like(x,dtype=float)
        chi2_dof = np.zeros_like(x,dtype=float)
        for i in range(len(x)):
            if ('mtype' in params) and ((params['mtype'] == 'Gsinh') or (params['mtype'] == 'Gexp') or (params['mtype'] == 'Gcosh') or (params['mtype'] == 'const')):
                E0_mean[i] = result[n][i*tau+tmin0][selected['tmax']]['mean'][0]
                E0_err[i] = result[n][i*tau+tmin0][selected['tmax']]['err'][0]
            else:
                E0_mean[i] = result[n][i*tau+tmin0][selected['tmax']]['mean'][1]
                E0_err[i] = result[n][i*tau+tmin0][selected['tmax']]['err'][1]
        if n == selected['n']:
            selected_time = (selected['tmin']-tmin0) // tau
            E0_err_selected_time = E0_err[selected_time]
            ylimup = E0_mean[selected_time] + E0_err[selected_time] * 8
            ylimdown = E0_mean[selected_time] - E0_err[selected_time] * 3


    for n in range(params['ns_min'],params['ns_max']+1):
        if params['mtype'] == 'const':
            if n != 1: continue
        tmin0 = params['tmin'][n][k][0]
        tmin1 = params['tmin'][n][k][1]
        tmin00 = min(tmin00, tmin0)
        tmin11 = max(tmin11, tmin1)
        x = np.arange(tmin0,tmin1+1,tau)
        E0_mean = np.zeros_like(x,dtype=float)
        E0_err = np.zeros_like(x,dtype=float)
        chi2_dof = np.zeros_like(x,dtype=float)
        for i in range(len(x)):
            if ('mtype' in params) and ((params['mtype'] == 'Gsinh') or (params['mtype'] == 'Gexp') or (params['mtype'] == 'Gcosh') or (params['mtype'] == 'const')):
                E0_mean[i] = result[n][i*tau+tmin0][selected['tmax']]['mean'][0]
                E0_err[i] = result[n][i*tau+tmin0][selected['tmax']]['err'][0]
            else:
                E0_mean[i] = result[n][i*tau+tmin0][selected['tmax']]['mean'][1]
                E0_err[i] = result[n][i*tau+tmin0][selected['tmax']]['err'][1]
        mask_large = E0_err > E0_err_selected_time * 3
        mask_normal = ~mask_large
        axes[0].errorbar(x=x[mask_normal]+0.1*(n-1),y=E0_mean[mask_normal],yerr=E0_err[mask_normal],mfc='none',color=inputs.clrs[n],marker=inputs.mrkr[n],alpha=inputs.alphas[n],linestyle='None',label=r'$n_s = %d$'%n,capsize=2)
        axes[0].errorbar(x=x[mask_large]+0.1*(n-1),y=E0_mean[mask_large],yerr=E0_err[mask_large],mfc='none',color=inputs.clrs[n],marker=inputs.mrkr[n],linestyle='None',capsize=2,alpha=0.1)
        '''if (n > 1):
            E1_mean = np.zeros_like(x,dtype=float)
            E1_err = np.zeros_like(x,dtype=float)
            for i in range(len(x)):
                E1_mean[i] = result[n][i+tmin0][selected['tmax']]['mean'][(fit_num+1)*1+fit_num]
                E1_err[i] = result[n][i+tmin0][selected['tmax']]['err'][(fit_num+1)*1+fit_num]
            axes[0].errorbar(x=x+0.1*(n-1),y=E1_mean,yerr=E1_err,mfc='none',color=inputs.clrs[n],marker=inputs.mrkr[n],alpha=inputs.alphas[n],linestyle='None',capsize=2)'''
        # chi2 plot
        for i in range(len(x)):
            chi2_dof[i] = result_chi[n][i*tau+tmin0][selected['tmax']]
        #axes[1].scatter(x=x+0.1*(n-1),y=chi2_dof,facecolor='none',edgecolor=inputs.clrs[n],marker=inputs.mrkr[n],zorder=10)
        axes[1].scatter(x=x+0.1*(n-1),y=chi2_dof,facecolor='w',edgecolor=inputs.clrs[n],marker=inputs.mrkr[n])
        if n == selected['n']:
            selected_time = (selected['tmin']-tmin0) // tau
            axes[0].scatter(selected['tmin']+0.1*(n-1),E0_mean[selected_time],color='k',marker=inputs.mrkr[n])
            axes[0].fill_between(np.array([0,params['T']/2+1]),E0_mean[selected_time]-E0_err[selected_time],E0_mean[selected_time]+E0_err[selected_time],color='y',alpha=0.2)
            '''if (n > 1):
                axes[0].scatter(selected['tmin']+0.1*(n-1),E1_mean[selected_time],color='k',marker=inputs.mrkr[n])
                axes[0].fill_between(np.array([0,tmin1+1]),E1_mean[selected_time]-E1_err[selected_time],E1_mean[selected_time]+E1_err[selected_time],color='y',alpha=0.2)'''
            axes[1].scatter(selected['tmin']+0.1*(n-1),chi2_dof[selected_time],color='k',marker=inputs.mrkr[n])

    # E plot
    if 'delta' in tit:
        axes[0].set_ylabel(r'$\delta E_n$')
    else:
        axes[0].set_ylabel(r'$E_n$')
    axes[0].axis([tmin00-1,tmin11+1,ylimdown,ylimup])
    # axes[0].set_xticks(np.arange(tmin00-1,tmin11+1))
    axes[0].tick_params(axis='x',bottom=True,direction='inout')
    axes[0].legend(ncol=params['ns_max'],columnspacing=0.5,loc=1)
    axes[0].set_title(tit)
    # chi2 plot
    axes[1].hlines(1,0,tmin11+1,linestyle=':',alpha=0.3,color='k')
    axes[1].axis([tmin00-1,tmin11+1,-0.1,2])
    axes[1].set_ylabel(r'$\chi^2/\mathrm{d.o.f.}$')
    axes[1].tick_params(axis='x',direction='in')
    axes[1].set_xlabel(r'$t_{\mathrm{min}}$')

    plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/stability_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    plt.show()


def plot_stability_ratio(k,fit_num,paramso,selectedo,result,result_chi,tit='',sv='',figdir=''):
    # Should use bootstraped result
    tau = paramso['tau']
    fig, axes = plt.subplots(2,1,sharex=True,gridspec_kw=dict(height_ratios=[4,1],hspace=0))
    params = dict(paramso)
    selected = dict(selectedo)
    relen = paramso['relen']
    selected['tmin'] = selectedo['tmin'][k]
    selected['tmax'] = selectedo['tmax'][k]
    selected['n'] = selectedo['n'][k]
    ylimup = 0
    ylimdown = 0
    tmin00 = 1000
    tmin11 = -1

    Gnum = int(k.split('_')[1])
    with open('../%s/spectra_full/%s/full_results_two_%s_%s_%d%d%d_%s_%s_%s.pckl'%(params['datadir']['mydata'],params['ensname'],params['ensname'],params['isospin'],params['P'][0],params['P'][1],params['P'][2],params['irrep'],k,params['tech']),'rb') as dfile:
        result_full_two = pickle.load(dfile)

    for n in range(params['ns_min'],params['ns_max']+1):
        tmin0 = params['tmin'][n][k][0]
        tmin1 = params['tmin'][n][k][1]
        tmin00 = min(tmin00, tmin0)
        tmin11 = max(tmin11, tmin1)
        x = np.arange(tmin0,tmin1+1,tau)
        E0_matrix = np.zeros((relen, len(x)), dtype=float)
        chi2_dof = np.zeros_like(x,dtype=float)
        for i in range(len(x)):
            for ls in range(relen):
                E0_matrix[ls][i] = result_full_two[n][i*tau+tmin0][selected['tmax']][ls].x[1] + 2 * params['pion_%d' % (Gnum)]['m'][ls]
        E0_mean = cal_mean(E0_matrix)
        E0_err = cal_err(E0_matrix, paramso['tech'])
        axes[0].errorbar(x=x+0.1*(n-1),y=E0_mean,yerr=E0_err,mfc='none',color=inputs.clrs[n],marker=inputs.mrkr[n],alpha=inputs.alphas[n],linestyle='None',label=r'$n_s = %d$'%n,capsize=2)
        '''if (n > 1):
            E1_mean = np.zeros_like(x,dtype=float)
            E1_err = np.zeros_like(x,dtype=float)
            for i in range(len(x)):
                E1_mean[i] = result[n][i+tmin0][selected['tmax']]['mean'][(fit_num+1)*1+fit_num]
                E1_err[i] = result[n][i+tmin0][selected['tmax']]['err'][(fit_num+1)*1+fit_num]
            axes[0].errorbar(x=x+0.1*(n-1),y=E1_mean,yerr=E1_err,mfc='none',color=inputs.clrs[n],marker=inputs.mrkr[n],alpha=inputs.alphas[n],linestyle='None',capsize=2)'''
        # chi2 plot
        for i in range(len(x)):
            chi2_dof[i] = result_chi[n][i*tau+tmin0][selected['tmax']]
        #axes[1].scatter(x=x+0.1*(n-1),y=chi2_dof,facecolor='none',edgecolor=inputs.clrs[n],marker=inputs.mrkr[n],zorder=10)
        axes[1].scatter(x=x+0.1*(n-1),y=chi2_dof,facecolor='w',edgecolor=inputs.clrs[n],marker=inputs.mrkr[n])
        if n == selected['n']:
            selected_time = (selected['tmin']-tmin0) // tau
            axes[0].scatter(selected['tmin']+0.1*(n-1),E0_mean[selected_time],color='k',marker=inputs.mrkr[n])
            axes[0].fill_between(np.array([0,params['T']/2+1]),E0_mean[selected_time]-E0_err[selected_time],E0_mean[selected_time]+E0_err[selected_time],color='y',alpha=0.2)
            '''if (n > 1):
                axes[0].scatter(selected['tmin']+0.1*(n-1),E1_mean[selected_time],color='k',marker=inputs.mrkr[n])
                axes[0].fill_between(np.array([0,tmin1+1]),E1_mean[selected_time]-E1_err[selected_time],E1_mean[selected_time]+E1_err[selected_time],color='y',alpha=0.2)'''
            axes[1].scatter(selected['tmin']+0.1*(n-1),chi2_dof[selected_time],color='k',marker=inputs.mrkr[n])
            
            ylimup = E0_mean[selected_time] + E0_err[selected_time] * 8
            ylimdown = E0_mean[selected_time] - E0_err[selected_time] * 3
    # E plot
    if 'delta' in tit:
        axes[0].set_ylabel(r'$\delta E_n$')
    else:
        axes[0].set_ylabel(r'$E_n$')
    axes[0].axis([tmin00-1,tmin11+1,ylimdown,ylimup])
    axes[0].set_xticks(np.arange(tmin00-1,tmin11+1))
    axes[0].tick_params(axis='x',bottom=True,direction='inout')
    axes[0].legend(ncol=params['ns_max'],columnspacing=0.5,loc=1)
    axes[0].set_title(tit)
    # chi2 plot
    axes[1].hlines(1,0,tmin11+1,linestyle=':',alpha=0.3,color='k')
    axes[1].axis([tmin00-1,tmin11+1,-0.1,4.1])
    axes[1].set_ylabel(r'$\chi^2/\mathrm{d.o.f.}$')
    axes[1].tick_params(axis='x',direction='in')
    axes[1].set_xlabel(r'$t_{\mathrm{min}}$')

    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/stability_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    plt.show()


def fit_function(x,para,nstates,params,mtype):
    if (mtype == 'dispersion'):
        [m, Z] = para
        f = m + Z * x
    # four-point and two-point
    if (mtype == 'const'):
        [m] = para
        f = m
    elif (nstates == 1) and (mtype == 'Gsinh'): # GEVPed subtracted correlator
        [E0, C] = para
        # Bad for I add C here.
        f = np.sinh(E0*(x-params['T']/2+1/2)) / np.sinh(E0*(params['t0']-params['T']/2+1/2))
    elif (nstates == 1) and (mtype == 'Gexp'):
        [E0, C] = para
        f = C * np.exp(-E0*(x-params['t0']))
    elif (nstates == 2) and (mtype == 'Gexp'):
        [E0, A, Ep] = para
        if (E0 > Ep):
            E0,Ep = Ep,E0
        f = (1-A) * np.exp(-E0*(x-params['t0'])) + A * np.exp(-Ep*(x-params['t0']))
    elif (nstates == 1) and (mtype == 'Gcosh'):
        [E0] = para
        # 2025.2.1: correct the fit form
        # f = np.cosh(E0/2*(params['T']-(x-params['t0']))) / np.cosh(E0/2*(params['T']))
        t0 = params['t0']
        T = params['T']
        f = (np.exp(-E0*(x-t0)) + np.exp(-E0*(T-(x-t0)))) / (1 + np.exp(-E0*T))
    elif (nstates == 2) and (mtype == 'Gcosh'):
        [E0, A, Ep] = para
        if (E0 > Ep):
            E0,Ep = Ep,E0
        # 2025.2.1: correct the fit form
        # f = (1-A) * np.cosh(E0/2*(params['T']-(x-params['t0']))) / np.cosh(E0/2*(params['T'])) + A * np.cosh(Ep/2*(params['T']-(x-params['t0']))) / np.cosh(Ep/2*(params['T']))
        t0 = params['t0']
        T = params['T']
        f = (1-A) * (np.exp(-E0*(x-t0)) + np.exp(-E0*(T-(x-t0)))) / (1 + np.exp(-E0*T)) + A * (np.exp(-Ep*(x-t0)) + np.exp(-Ep*(T-(x-t0)))) / (1 + np.exp(-Ep*T))
    elif nstates == 1:
        [A0, E0] = para
        if (mtype == 'exp'):
            f = A0 * np.exp(-E0*x)
        elif (mtype == 'cosh'): # normal correlator
            f = A0 * (np.exp(-E0*x)+np.exp(-E0*(params['T']-x)))
        elif (mtype == 'sinh'): # subtracted correlator
            f = 4 * A0 * np.exp(-E0*params['T']/2) * np.sinh(E0/2) * np.sinh(E0*(x-params['T']/2+1/2))
        elif (mtype == 'R'):
            dE = E0
            f = A0 * np.exp(-dE*params['T']/2)*((np.cosh((2*params['m']+dE)*(params['T']/2-x))-np.cosh((2*params['m']+dE)*(params['T']/2-x-1)))/(np.cosh(params['m']*(params['T']/2-x))**2-np.cosh(params['m']*(params['T']/2-x-1))**2))
        elif (mtype == 'S3'):
            dE = E0
            f = A0 * (np.cosh((3*params['m']+dE)*(params['T']/2-x))/np.cosh(params['m']*(params['T']/2-x))-np.cosh((3*params['m']+dE)*(params['T']/2-x-1))/np.cosh(params['m']*(params['T']/2-x-1)))
        elif (mtype == 'R3'):
            dE = E0
            f = A0 * ((np.cosh((3*params['m']+dE)*(x-params['T']/2))/np.cosh(params['m']*(x-params['T']/2))-np.cosh((3*params['m']+dE)*(x-params['T']/2+1))/np.cosh(params['m']*(x-params['T']/2+1)))/(np.sinh(params['m'])*np.sinh(params['m']*(1+2*x-params['T']))))
    elif nstates == 2:
        [A0, E0, A1, E1] = para
        # 2022.10.12
        if (E0 > E1):
            E0,E1 = E1,E0
            A0,A1 = A1,A0
        if (mtype == 'exp'):
            f = A0 * np.exp(-E0*x) + A1 * np.exp(-E1*x)
        elif (mtype == 'cosh'): # normal correlator
            f = A0 * (np.exp(-E0*x)+np.exp(-E0*(params['T']-x))) + A1 * (np.exp(-E1*x)+np.exp(-E1*(params['T']-x)))
        elif (mtype == 'sinh'): # subtracted correlator
            f = 4 * A0 * np.exp(-E0*params['T']/2) * np.sinh(E0/2) * np.sinh(E0*(x-params['T']/2+1/2)) + 4 * A1 * np.exp(-E1*params['T']/2) * np.sinh(E1/2) * np.sinh(E1*(x-params['T']/2+1/2))
    return f


def plot_result(data,paramso,selectedo,result,ans,tau=1,mtype='exp',cylim='no',tit='',sv='',figdir=''):
    # 2023.03.23: Use only the good result
    # Use all data and all result
    # fig, axes = plt.subplots(2,1,sharex=True,gridspec_kw=dict(hspace=0))
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(6.4, 4)
    relen = paramso['relen']
    data_mean = {}
    data_err = {}
    meff = {}
    meff_mean = {}
    meff_err = {}
    dt = 0.01
    params = dict(paramso)
    selected = dict(selectedo)
    for k in data.keys():
        selected['tmin'] = selectedo['tmin'][k]
        selected['tmax'] = selectedo['tmax'][k]
        selected['n'] = selectedo['n'][k]
    for i,k in enumerate(data.keys()):
        # Data
        data_mean[k] = cal_mean(data[k])
        data_err[k] = cal_err(data[k],params['tech'])
        x = np.arange(data[k].shape[1])
        meff[k] = np.zeros_like(data[k])
        for j in range(data[k].shape[0]):
            meff[k][j] = cal_mass(data[k][j],mtype=mtype,tau=tau)
        meff_mean[k] = cal_mean(meff[k])
        meff_err[k] = cal_err(meff[k],params['tech'])
        # axes[0].errorbar(x=x,y=data_mean[k],yerr=data_err[k],ls='None',marker='o',color=inputs.clrs[i],mec=inputs.clrs[i],capsize=2,fillstyle='none',label=inputs.labels(k))
        fit_mask = (x >= selected['tmin']) & (x <= selected['tmax'])
        axes.errorbar(x=x[~fit_mask],y=meff_mean[k][~fit_mask],yerr=meff_err[k][~fit_mask],ls='None',marker='o',color=inputs.clrs[i],mec=inputs.clrs[i],capsize=2,fillstyle='none',alpha=0.1)
        axes.errorbar(x=x[fit_mask],y=meff_mean[k][fit_mask],yerr=meff_err[k][fit_mask],ls='None',marker='o',color=inputs.clrs[i],mec=inputs.clrs[i],capsize=2,fillstyle='none')

        # Reconstructed
        xx = np.arange(selected['tmin'],selected['tmax'],dt)
        recon_corr_matrix = np.zeros([relen,len(xx)])
        recon_meff_matrix = np.zeros([relen,len(xx)])
        for ls in range(relen):
            result_para = result[ls]
            recon_corr_matrix[ls] = fit_function(xx,result_para,selected['n'],params,mtype)
            if (mtype == 'exp') or (mtype == 'Gexp'):
                recon_meff_matrix[ls] = 1./dt * np.log(recon_corr_matrix[ls] / np.roll(recon_corr_matrix[ls],-1))
            elif (mtype == 'cosh') or (mtype == 'sinh') or (mtype == 'Gsinh') or (mtype == 'Gcosh'):
                recon_meff_matrix[ls] = 1./dt * np.arccosh((np.roll(recon_corr_matrix[ls],-1)+np.roll(recon_corr_matrix[ls],1))/2/recon_corr_matrix[ls])
            elif (mtype =='const'):
                recon_meff_matrix[ls] = fit_function(xx,result_para,selected['n'],params,mtype)
                recon_corr_matrix[ls] = np.exp(-recon_meff_matrix[ls] * xx)
                recon_corr_matrix[ls] = np.exp(-recon_meff_matrix[ls] * xx) * data_mean[k][selected['tmin']] * np.exp(recon_meff_matrix[ls][0] * selected['tmin'])
        recon_corr_mean = cal_mean(recon_corr_matrix)
        recon_corr_err = cal_err(recon_corr_matrix,tech=params['tech'])
        recon_meff_mean = cal_mean(recon_meff_matrix)
        recon_meff_err = cal_err(recon_meff_matrix,tech=params['tech'])
        #axes[0].plot(xx,recon_corr_mean-recon_corr_err,color=inputs.clrs[i],alpha=0.3)
        #axes[0].plot(xx,recon_corr_mean+recon_corr_err,color=inputs.clrs[i],alpha=0.3)
        # axes[0].fill_between(xx,recon_corr_mean-recon_corr_err,recon_corr_mean+recon_corr_err,color=inputs.clrs[i],alpha=0.3,edgecolor='none')

        xx = xx[:-1]
        recon_meff_mean = recon_meff_mean[:-1]
        recon_meff_err = recon_meff_err[:-1]
        #axes[1].plot(xx,recon_meff_mean-recon_meff_err,color=inputs.clrs[i],alpha=0.3)
        #axes[1].plot(xx,recon_meff_mean+recon_meff_err,color=inputs.clrs[i],alpha=0.3)
        axes.fill_between(xx,recon_meff_mean-recon_meff_err,recon_meff_mean+recon_meff_err,color=inputs.clrs[i],alpha=0.3,edgecolor='none')

        # Result value
        if (mtype != 'Gsinh') and (mtype != 'Gexp') and (mtype != 'Gcosh') and (mtype != 'const'):
            #axes[1].hlines(ans['mean'][len(data)],0,params['T']/2+1.5,linestyle=':',alpha=0.4,color='k')
            axes.fill_between(np.array([selected['tmin'],selected['tmax']]),ans['mean'][len(data)]-ans['err'][len(data)],ans['mean'][len(data)]+ans['err'][len(data)],color='gray',alpha=0.4,edgecolor='none')
            axes.fill_between(np.array([0,selected['tmin']]),ans['mean'][len(data)]-ans['err'][len(data)],ans['mean'][len(data)]+ans['err'][len(data)],color='gray',alpha=0.2,edgecolor='none')
            axes.fill_between(np.array([selected['tmax'],params['T']/2+1.5]),ans['mean'][len(data)]-ans['err'][len(data)],ans['mean'][len(data)]+ans['err'][len(data)],color='gray',alpha=0.2,edgecolor='none')
        else:
            #axes[1].hlines(ans['mean'][0],0,params['T']/2+1.5,linestyle=':',alpha=0.4,color='k')
            axes.fill_between(np.array([selected['tmin'],selected['tmax']]),ans['mean'][0]-ans['err'][0],ans['mean'][0]+ans['err'][0],color='gray',alpha=0.4,edgecolor='none')
            axes.fill_between(np.array([0,selected['tmin']]),ans['mean'][0]-ans['err'][0],ans['mean'][0]+ans['err'][0],color='gray',alpha=0.2,edgecolor='none')
            axes.fill_between(np.array([selected['tmax'],params['T']/2+1.5]),ans['mean'][0]-ans['err'][0],ans['mean'][0]+ans['err'][0],color='gray',alpha=0.2,edgecolor='none')
    axes.set_title(tit)
    axes.set_xlim([0,params['T']/2+1.5])
    # axes.set_xlim([0,30])
    # axes[0].set_ylim([np.nanmin(data_mean[k]) - data_err[k][np.nanargmin(data_mean[k])] * 10,np.nanmax(data_mean[k]) + data_err[k][np.nanargmax(data_mean[k])] * 10])
    # if cylim != 'no':
    #     axes[0].set_ylim(cylim)
    # axes[0].set_ylim([recon_corr_mean[-1]-recon_corr_err[-1],recon_corr_mean[0]+recon_corr_err[0]])
    if (mtype != 'Gsinh') and (mtype != 'Gexp') and (mtype != 'Gcosh') and (mtype != 'const'):
        axes.set_ylim([ans['mean'][len(data)]-ans['err'][len(data)]*3,ans['mean'][len(data)]+ans['err'][len(data)]*10])
    else:
        axes.set_ylim([ans['mean'][0]-ans['err'][0]*9,ans['mean'][0]+ans['err'][0]*30])
    axes.set_xlabel(r'$t$')
    # axes[0].set_ylabel(r'$C(t)$')
    axes.set_ylabel(r'$m_{\mathrm{eff}}$')
    # axes.legend()
    #axes[0].set_yscale('symlog',linthresh=1e-30)
    # if np.all(recon_corr_mean > 0):
    #     axes[0].set_yscale('log')
    plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/result_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    plt.show()


def GEVP_sort(method, corrmat_ls, Gseries_list, v_list, t0, tref):
    # Only one sample
    N = len(Gseries_list[0])
    T = len(Gseries_list)

    if method == 'value':
        for t in range(T):
            sort_ind = np.argsort(Gseries_list[t])
            Gseries_list[t] = Gseries_list[t][sort_ind]
            v_list[t] = v_list[t][:,sort_ind]
            if t > t0:
                Gseries_list[t] = Gseries_list[t][::-1]
                v_list[t] = v_list[t][:,::-1]

    elif method == 'vector':
        refv = v_list[tref][:,np.argsort(Gseries_list[tref])]
        if tref > t0:
            refv = refv[:,::-1]
        for t in range(T):
            state_args = np.zeros(N, dtype=int) - 1
            for refstate in range(N):
                state_overlaps = np.zeros(N, dtype=complex)
                for state in range(N):
                    state_overlaps[state] = np.dot(np.conjugate(refv[:,refstate]), np.dot(corrmat_ls[:,:,t0], v_list[t][:,state]))
                # From largest to smallest
                overlap_sorted_ind = np.argsort(-np.abs(state_overlaps))
                for i in range(N):
                    if overlap_sorted_ind[i] not in state_args:
                        state_args[refstate] = overlap_sorted_ind[i]
                        break
                # state_args[refstate] = overlap_sorted_ind[0]
            Gseries_list[t] = Gseries_list[t][state_args]
            v_list[t] = v_list[t][:,state_args]

    elif method == 'vector_repeat':
        refv = v_list[tref][:,np.argsort(Gseries_list[tref])]
        if tref > t0:
            refv = refv[:,::-1]
        for t in range(T):
            state_args = np.zeros(N, dtype=int)
            for refstate in range(N):
                state_overlaps = np.zeros(N, dtype=complex)
                for state in range(N):
                    state_overlaps[state] = np.dot(np.conjugate(refv[:,refstate]), np.dot(corrmat_ls[:,:,t0], v_list[t][:,state]))
                state_args[refstate] = np.argmax(np.abs(state_overlaps))
            Gseries_list[t] = Gseries_list[t][state_args]
            v_list[t] = v_list[t][:,state_args]

    return Gseries_list, v_list


def GEVP(corrmat, method, key, t0, tref):
    # Source of evil
    #corrmat = corrmat.real
    N = corrmat.shape[1]
    relen = corrmat.shape[0]
    T = corrmat.shape[3]
    Gfdata = {}
    for i in range(N):
        Gfdata['%s_%d' % (key, i)] = np.zeros([relen, T])
    vecs = np.zeros([relen, T, N, N], dtype=complex)

    for ls in range(relen):
        Gseries_list = []
        v_list = []
        for t in range(T):
            [Gseries, v] = linalg.eig(corrmat[ls,:,:,t],corrmat[ls,:,:,t0])
            Gseries_list.append(Gseries)
            v_list.append(v)

        [Gseries_list, v_list] = GEVP_sort(method, corrmat[ls], Gseries_list, v_list, t0, tref)

        for t in range(T):
            for i in range(N):
                Gfdata['%s_%d' % (key, i)][ls,t] = np.copy(Gseries_list[t][i].real)

        vecs[ls] = np.array(v_list)

    return Gfdata, vecs


def chi(para,data,Linv,tmin,tmax,nstates,params,mtype):
    tau = params['tau']
    chi_now = np.array([])
    for j,k in enumerate(data.keys()):
        x = np.arange(tmin,tmax+1,tau)
        f = 0
        # four-point and two-point
        f = fit_function(x,para,nstates,params,mtype)
        row = (data[k][x] - f)
        chi_now = np.concatenate((chi_now,np.dot(Linv[k],row)))
    return chi_now


def chi_dispersion(para,data,Linv,params,x,mtype):
    # Mean data
    chi_now = np.array([])
    for j,k in enumerate(data.keys()):
        f = 0
        # four-point and two-point
        f = fit_function(x,para,0,params,mtype)
        row = (data[k] - f)
        chi_now = np.concatenate((chi_now,np.dot(Linv[k],row)))
    return chi_now


def sort_fit(fit, params):
    nstates = params['nstates']
    mtype = params['mtype']
    #prior = params['prior']
    para = fit.x
    if (mtype == 'const'):
        return
    if (nstates == 2) and (mtype == 'Gexp'):
        [E0, A, Ep] = para
        #[E0_prior, _, Ep_prior] = prior
        E0_prior = 0
        if abs(E0-E0_prior) > abs(Ep-E0_prior):
            fit.x = np.array([Ep, A, E0])
    elif (nstates == 2) and (mtype == 'Gcosh'):
        [E0, A, Ep] = para
        #[E0_prior, _, Ep_prior] = prior
        E0_prior = 0
        if abs(E0-E0_prior) > abs(Ep-E0_prior):
            fit.x = np.array([Ep, A, E0])
    elif nstates == 2:
        [A0, E0, A1, E1] = para
        #[_, E0_prior, _, E1_prior] = prior
        E0_prior = 0
        if abs(E0-E0_prior) > abs(E1-E0_prior):
            fit.x = np.array([A1, E1, A0, E0])


def mpfitting(ls, params, mtype, k, data, prior, Linv, tmin, tmax, n):
    if mtype == 'R':
        params['m'] = params['single']['m'][ls]
    redata = {}
    redata[k] = data[k][ls]
    if n == 1:
        fit = least_squares(chi,prior,args=(redata,Linv,tmin,tmax,n,params,mtype))
    elif n == 2:
        #fit = least_squares(chi,prior,args=(redata,Linv,tmin,tmax,n,params,mtype),bounds=([0,0,0,0],[500,5,500,10]))
        fit = least_squares(chi,prior,args=(redata,Linv,tmin,tmax,n,params,mtype))
    sort_fit(fit, params)
    return fit


def fitting(p,paramso,data,mtype,selectedo,correlated=True):
    # Fitting
    # Initialize the result dictionary
    # result: saves all results from resampling fit
    # result_para: results of parameters after calculating errors, with means coming from result_mean
    # result_chi2dof: results of chi^2/d.o.f, from result_mean
    def gen_prior(p, n, mtype):
        if (mtype == 'const'):
            prior = [p['E_0']]
        elif n == 1 and ((mtype == 'Gsinh') or (mtype == 'Gexp')):
            prior = [p['E_0'],1]
        elif n == 2 and mtype == 'Gexp':
            prior = [p['E_0'],p['A'],p['E_p']]
        elif n == 1 and mtype == 'Gcosh':
            prior = [p['E_0']]
        elif n == 2 and mtype == 'Gcosh':
            prior = [p['E_0'],p['A'],p['E_p']]
        elif n == 1:
            prior = [p['A_0']
                    ,p['E_0']]
        elif n == 2:
            prior = [p['A_0'],p['E_0'],p['A_1'],p['E_1']]
        return prior

    relen = paramso['relen']
    tau = paramso['tau']
    result = {}
    result_para = {}
    result_chi2dof = {}
    params = dict(paramso)
    selected = dict(selectedo)
    # Only one key
    k = list(data.keys())[0]
    selected['tmin'] = selectedo['tmin'][k]
    selected['tmax'] = selectedo['tmax'][k]
    selected['n'] = selectedo['n'][k]

    for n in range(params['ns_min'],params['ns_max']+1):
        result[n] = {}
        result_para[n] = {}
        result_chi2dof[n] = {}
        for tmin in range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1):
            result[n][tmin] = {}
            result_para[n][tmin] = {}
            result_chi2dof[n][tmin] = {}
            for tmax in range(params['tmax'][n][k][0],params['tmax'][n][k][1]+1):
                result[n][tmin][tmax] = {}
                result_para[n][tmin][tmax] = {}

    if not paramso['just_changing_tmin']:
        # Cov matrix
        cov = {}
        for i,k in enumerate(data.keys()):
            cov[k] = cal_cov(data[k], params['tech'])
            if not correlated:
                cov_diag = np.diag(cov[k])
                cov_diag = np.where(cov_diag == 0, 1e-32, cov_diag)
                cov[k] = np.diag(cov_diag)
        # Fitting
        for n in range(params['ns_min'],params['ns_max']+1):
            params['nstates'] = n
            prior = gen_prior(p, n, mtype)
            for i,k in enumerate(data.keys()):
                #for tmin in tqdm.tqdm(range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1),desc='Fit: %s'%(k)):
                for tmin in range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1,tau):
                    for tmax in range(params['tmax'][n][k][0],params['tmax'][n][k][1]+1,tau):
                        Linv = {}
                        Linv[k] = np.linalg.inv(np.linalg.cholesky(cov[k][tmin:tmax+1:tau,tmin:tmax+1:tau]))
                        # Resample fit
                        with Pool(processes = os.cpu_count()) as pool:
                            pool_result = pool.starmap(mpfitting, [(ls, params, mtype, k, data, prior, Linv, tmin, tmax, n) for ls in range(relen)])
                        for ls in range(relen):
                            result[n][tmin][tmax][ls] = pool_result[ls]
        # Saving purly for just changing tmin
        dfile = open('../%s/spectra_full/%s/full_results_2pt_%s_%s_%s.pckl'%(params['datadir']['mydata'],params['ensemble'],params['ensemble'],k,params['tech']),'wb')
        pickle.dump(result,dfile)
        dfile.close()
    else:
        dfile = open('../%s/spectra_full/%s/full_results_2pt_%s_%s_%s.pckl'%(params['datadir']['mydata'],params['ensemble'],params['ensemble'],k,params['tech']),'rb')
        result = pickle.load(dfile)
        dfile.close()

    # Bootstrap the fitting results
    for n in range(params['ns_min'],params['ns_max']+1):
        params['nstates'] = n
        prior = gen_prior(p, n, mtype)
        for i,k in enumerate(data.keys()):
            for tmin in range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1,tau):
                for tmax in range(params['tmax'][n][k][0],params['tmax'][n][k][1]+1,tau):
                    para_matrix = np.zeros([relen,len(prior)])
                    for ls in range(relen):
                        fit = result[n][tmin][tmax][ls]
                        para_matrix[ls]=fit.x
                    result_para[n][tmin][tmax]['mean'] = cal_mean(para_matrix)
                    result_para[n][tmin][tmax]['err'] = cal_err(para_matrix, tech=params['tech'])
                    result_chi2dof[n][tmin][tmax] = np.sum(result[n][tmin][tmax][0].fun**2)/(tmax-tmin+1-len(prior))

    # Set a lazy tmin selection
    selected_lazy_tmin = -1
    n = selected['n']
    if paramso['lazy_tmin'] == True:
        chi2_score = 10000
        has_good_fit = False
        for tmin in range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1,tau):
            err = result_para[n][tmin][tmax]['err'][0]
            if err < 0.005:
                has_good_fit = True
                break
        for tmin in range(params['tmin'][n][k][0],params['tmin'][n][k][1]+1,tau):
            err = result_para[n][tmin][tmax]['err'][0]
            if err >= 0.005 and has_good_fit:
                continue
            chi2dof = result_chi2dof[n][tmin][tmax]
            score_now = 2 * np.log(np.abs(chi2dof - 0.9)) + np.log(err)
            if score_now < chi2_score:
                chi2_score = score_now
                selected_lazy_tmin = tmin

    selected_use_tmin = selected_lazy_tmin if paramso['lazy_tmin'] == True else selected['tmin']

    # The results
    ans = result_para[selected['n']][selected_use_tmin][selected['tmax']]
    #ans_chi = result_chi2dof[selected['n']][selected_use_tmin][selected['tmax']]

    result_good = {}
    for ls in range(relen):
        result_good[ls] = result[selected['n']][selected_use_tmin][selected['tmax']][ls].x

    return selected_lazy_tmin, result_para, result_chi2dof, result_good, ans


def plot_stability_3pt(k,paramso,selectedo,result,result_chi,tit='',sv='',figdir=''):
    tau = paramso['tau']
    fig, axes = plt.subplots(2,1,sharex=True,gridspec_kw=dict(height_ratios=[4,1],hspace=0))
    fig.set_size_inches(6.4, 4)
    params = dict(paramso)
    n = selectedo['n'][k]
    tins0 = min(min(result[tsep][n]) for tsep in params['tsep_list'])
    tins1 = max(max(result[tsep][n]) for tsep in params['tsep_list'])
    selected_low = []
    selected_up = []
    for itsep, tsep in enumerate(params['tsep_list']):
        x = np.array(sorted(result[tsep][n]))
        tins = selectedo['tins'][k][tsep]
        mean = np.array([result[tsep][n][t]['mean'][0] for t in x])
        err = np.array([result[tsep][n][t]['err'][0] for t in x])
        chi2 = np.array([result_chi[tsep][n][t] for t in x])
        selected_time = np.where(x == tins)[0][0]
        selected_low.append(mean[selected_time] - 3 * err[selected_time])
        selected_up.append(mean[selected_time] + 8 * err[selected_time])
        color = inputs.clrscm(len(params['tsep_list']),itsep)
        axes[0].errorbar(x=x+0.1*itsep,y=mean,yerr=err,mfc='none',color=color,marker='o',linestyle='None',label=r'$t_{sep} = %d$'%tsep,capsize=2)
        axes[1].scatter(x=x+0.1*itsep,y=chi2,facecolor='w',edgecolor=color,marker='o')
        axes[0].scatter(tins+0.1*itsep,mean[selected_time],color='k',marker='o')
        axes[1].scatter(tins+0.1*itsep,chi2[selected_time],color='k',marker='o')
    axes[0].set_ylabel(r'$C$')
    axes[0].set_ylim([min(selected_low),max(selected_up)])
    axes[0].set_xlim([tins0-1,tins1+1])
    axes[0].legend(ncol=len(params['tsep_list']),columnspacing=0.5,loc=1)
    axes[0].set_title(tit)
    axes[1].hlines(1,tins0-1,tins1+1,linestyle=':',alpha=0.3,color='k')
    axes[1].set_ylim([-0.1,2])
    axes[1].set_ylabel(r'$\chi^2/\mathrm{d.o.f.}$')
    axes[1].set_xlabel(r'$t_{\mathrm{ins}}$')

    plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/stability_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    # show_in_spyder()


def plot_result_3pt(k,data,paramso,selectedo,result,tit='',sv='',figdir=''):
    fig, axes = plt.subplots(1,2,sharey=True,gridspec_kw={'width_ratios':[5,0.6],'wspace':0.03})
    fig.set_size_inches(7.2, 4)
    ax = axes[0]
    ax_tsep = axes[1]
    relen = paramso['relen']
    params = dict(paramso)
    tech = params['tech']
    selected = dict(selectedo)
    selected['n'] = selectedo['n'][k]
    moment = params['moment']
    dt = 0.01
    ymin = []
    ymax = []
    xlim = [-max(params['tsep_list']) // 2 - 1,max(params['tsep_list']) // 2 + 1]
    for itsep, tsep in enumerate(params['tsep_list']):
        x = np.arange(tsep+1) - tsep // 2
        color = inputs.clrscm(len(params['tsep_list']),itsep)
        data_mean = cal_mean(data[tsep])
        data_err = cal_err(data[tsep],tech)
        tins = selectedo['tins'][k][tsep]
        fit_mask = np.abs(x) <= tins
        ax.errorbar(x=x[~fit_mask],y=data_mean[~fit_mask],yerr=data_err[~fit_mask],ls='None',marker='o',color=color,mec=color,capsize=2,fillstyle='none',alpha=0.1)
        ax.errorbar(x=x[fit_mask],y=data_mean[fit_mask],yerr=data_err[fit_mask],ls='None',marker='o',color=color,mec=color,capsize=2,fillstyle='none',label=r'$t_{sep} = %d$'%tsep)
        xx = np.arange(xlim[0],xlim[1] + dt,dt)
        recon_matrix = np.zeros([relen,len(xx)])
        for ls in range(relen):
            recon_matrix[ls] = fit_function(xx,result[tsep][ls],moment,params,params['mtype'])
        recon_mean = cal_mean(recon_matrix)
        recon_err = cal_err(recon_matrix,tech)
        fit_curve = np.abs(xx) <= tins
        ax.fill_between(xx,recon_mean-recon_err,recon_mean+recon_err,color=color,alpha=0.1,edgecolor='none')
        ax.fill_between(xx[fit_curve],recon_mean[fit_curve]-recon_err[fit_curve],recon_mean[fit_curve]+recon_err[fit_curve],color=color,alpha=0.3,edgecolor='none')
        fit_value = np.array([np.ravel(result[tsep][ls])[0] for ls in range(relen)])
        fit_mean = cal_mean(fit_value)
        fit_err = cal_err(fit_value,tech)
        ymin.append(fit_mean-fit_err*10)
        ymax.append(fit_mean+fit_err*10)
        ax_tsep.errorbar(tsep,fit_mean,yerr=fit_err,ls='None',marker='o',color=color,mec=color,capsize=2,fillstyle='none')
    ax.set_ylim([min(ymin),max(ymax)])
    ax.set_title(tit)
    ax.set_xlim(xlim)
    ax.legend(ncol=len(params['tsep_list']),columnspacing=0.5,loc=1)
    ax.set_xlabel(r'$t_j - t_{sep}/2$')
    ax.set_ylabel(r'$\langle x^%d \rangle / \langle x \rangle$' % (moment - 1))
    ax_tsep.set_xlabel(r'$t_{sep}$')
    ax_tsep.set_xticks(params['tsep_list'])
    ax_tsep.set_xlim([min(params['tsep_list'])-2,max(params['tsep_list'])+2])
    ax_tsep.tick_params(axis='x',labelrotation=0)
    plt.tight_layout()
    Path('../%s/%s/'%(params['figures'],figdir)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../%s/%s/result_%s.pdf'%(params['figures'],figdir,sv),transparent=True)
    # show_in_spyder()


def chi_3pt(para,data,Linv,tins,nstates,params,mtype):
    tau = params['tau']
    chi_now = np.array([])
    for j,k in enumerate(data.keys()):
        x = np.arange(len(data[k])//2-tins,len(data[k])//2+tins+1,tau)
        f = fit_function(x,para,nstates,params,mtype)
        row = (data[k][x] - f)
        chi_now = np.concatenate((chi_now,np.dot(Linv[k],row)))
    return chi_now


def mpfitting_3pt(ls, params, mtype, k, data, prior, Linv, tins, n):
    redata = {}
    redata[k] = data[k][ls]
    fit = least_squares(chi_3pt,prior,args=(redata,Linv,tins,n,params,mtype))
    return fit


def fitting_3pt(p,paramso,data,mtype,selectedo,correlated=True):
    def gen_prior(p, n, mtype):
        if (mtype == 'const'):
            prior = [p['C']]
        return prior

    relen = paramso['relen']
    tau = paramso['tau']
    result = {}
    result_chi2dof = {}
    params = dict(paramso)
    selected = dict(selectedo)
    # Only one key
    k = list(data.keys())[0]
    selected['tins'] = selectedo['tins'][k]
    selected['n'] = selectedo['n'][k]

    for n in [selected['n']]:
        result[n] = {}
        result_chi2dof[n] = {}
        for tins in range(params['tins'][n][k][0],params['tins'][n][k][1]+1,tau):
            result[n][tins] = {}
            result_chi2dof[n][tins] = {}

    if not paramso['just_changing_tins']:
        # Cov matrix
        cov = {}
        for i,k in enumerate(data.keys()):
            cov[k] = cal_cov(data[k], params['tech'])
            if not correlated:
                cov_diag = np.diag(cov[k])
                cov_diag = np.where(cov_diag == 0, 1e-32, cov_diag)
                cov[k] = np.diag(cov_diag)
        # Fitting
        for n in [selected['n']]:
            params['nstates'] = n
            prior = gen_prior(p, n, mtype)
            for i,k in enumerate(data.keys()):
                for tins in range(params['tins'][n][k][0],params['tins'][n][k][1]+1,tau):
                    if tins == 0:
                        middle = len(data[k][0]) // 2
                        for ls in range(relen):
                            result[n][tins][ls] = np.array([data[k][ls,middle]])
                    else:
                        middle = len(data[k][0]) // 2
                        time = slice(middle-tins,middle+tins+1,tau)
                        Linv = {}
                        Linv[k] = np.linalg.inv(np.linalg.cholesky(cov[k][time,time]))
                        if mtype == 'const':
                            # For R(t)=C, minimize ||L^{-1}(y-C)||^2 for
                            # each resample.  This is the correlated GLS
                            # solution and replaces the 1-parameter optimizer.
                            sample_data = np.asarray(data[k][:, time])
                            ones = np.ones(sample_data.shape[1])
                            whitened_data = sample_data @ Linv[k].T
                            whitened_ones = ones @ Linv[k].T
                            denominator = np.dot(whitened_ones, whitened_ones)
                            fit_values = np.dot(whitened_data, whitened_ones) / denominator
                            fit_residuals = (sample_data - fit_values[:, None]) @ Linv[k].T
                            pool_result = []
                            for ls in range(relen):
                                fit = OptimizeResult()
                                fit.x = np.array([fit_values[ls]])
                                fit.fun = fit_residuals[ls]
                                fit.cost = 0.5 * np.dot(fit.fun, fit.fun)
                                fit.nfev = 0
                                fit.success = True
                                fit.status = 1
                                fit.message = 'analytic correlated GLS fit'
                                pool_result.append(fit)
                        else:
                            with Pool(processes = os.cpu_count()) as pool:
                                pool_result = pool.starmap(mpfitting_3pt, [(ls, params, mtype, k, data, prior, Linv, tins, n) for ls in range(relen)])
                        for ls in range(relen):
                            result[n][tins][ls] = pool_result[ls]
        # Saving purly for just changing tins
        dfile = open('../%s/spectra_full/%s/full_results_3pt_%s_%s_n%d_tf%d_tsep%d_%s.pckl'%(params['datadir']['mydata'],params['ensemble'],params['ensemble'],k,n,params['tf'],params['tsep'],params['tech']),'wb')
        pickle.dump(result,dfile)
        dfile.close()
    else:
        print('    reading cached 3pt fits: n=%d, tf=%d, tsep=%d' % (n,params['tf'],params['tsep']), flush=True)
        dfile = open('../%s/spectra_full/%s/full_results_3pt_%s_%s_n%d_tf%d_tsep%d_%s.pckl'%(params['datadir']['mydata'],params['ensemble'],params['ensemble'],k,n,params['tf'],params['tsep'],params['tech']),'rb')
        result = pickle.load(dfile)
        dfile.close()

    for n in [selected['n']]:
        prior = gen_prior(p, n, mtype)
        for tins in result[n]:
            if tins == 0:
                result_chi2dof[n][tins] = 0
            else:
                result_chi2dof[n][tins] = np.sum(result[n][tins][0].fun**2)/(2*tins+1-len(prior))
        if 'stability' in params:
            params['stability'][k] = {n: {}}
            for tins in result[n]:
                para_matrix = np.array([result[n][tins][ls] if tins == 0 else result[n][tins][ls].x for ls in range(relen)])
                params['stability'][k][n][tins] = {'mean': cal_mean(para_matrix), 'err': cal_err(para_matrix,tech=params['tech'])}

    # Set a lazy tins selection
    selected_lazy_tins = -1
    n = selected['n']
    if paramso['lazy_tins'] == True:
        chi2_score = 10000
        has_good_fit = False
        for tins in range(params['tins'][n][k][0],params['tins'][n][k][1]+1,tau):
            para_matrix = np.array([result[n][tins][ls] if tins == 0 else result[n][tins][ls].x for ls in range(relen)])
            err = cal_err(para_matrix,tech=params['tech'])[0]
            if err < 0.005:
                has_good_fit = True
                break
        for tins in range(params['tins'][n][k][0],params['tins'][n][k][1]+1,tau):
            para_matrix = np.array([result[n][tins][ls] if tins == 0 else result[n][tins][ls].x for ls in range(relen)])
            err = cal_err(para_matrix,tech=params['tech'])[0]
            if err >= 0.005 and has_good_fit:
                continue
            chi2dof = result_chi2dof[n][tins]
            score_now = 2 * np.log(np.abs(chi2dof - 0.9)) + np.log(err)
            if score_now < chi2_score:
                chi2_score = score_now
                selected_lazy_tins = tins

    selected_use_tins = selected_lazy_tins if paramso['lazy_tins'] == True else selected['tins']
    result_good = {}
    for ls in range(relen):
        result_good[ls] = result[n][selected_use_tins][ls] if selected_use_tins == 0 else result[n][selected_use_tins][ls].x

    return result_chi2dof, result_good
