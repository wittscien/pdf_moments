from pathlib import Path
import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import funcs as tp
import inputs


def fit_function(x, para):
    [a, b] = para
    return a + b * x


def chi(para, data, Linv, x):
    chi_now = np.array([])
    for k in data.keys():
        row = data[k] - fit_function(x,para)
        chi_now = np.concatenate((chi_now,np.dot(Linv[k],row)))
    return chi_now


def fitting(x, samples):
    tech = 'bootstrap'
    x = np.asarray(x, dtype=float)
    samples = np.asarray(samples, dtype=float)
    prior = [np.mean(samples[0]), 0.0]
    cov = tp.cal_cov(samples,tech)
    Linv = {'data': np.linalg.inv(np.linalg.cholesky(cov))}
    result = {}
    para_matrix = np.zeros([len(samples),len(prior)])
    for ls in range(len(samples)):
        redata = {'data': samples[ls]}
        result[ls] = least_squares(chi,prior,args=(redata,Linv,x))
        para_matrix[ls] = result[ls].x
    result_para = {}
    result_para['samples'] = para_matrix
    result_para['mean'] = tp.cal_mean(para_matrix)
    result_para['err'] = tp.cal_err(para_matrix,tech)
    result_para['chi2'] = np.sum(result[0].fun ** 2)
    result_para['dof'] = len(samples[0]) - len(prior)
    return result_para


def continuum_extrapolation(params, metadata, data, ensembles, fit_range, figdir):
    tech = 'bootstrap'
    data_color = '#3C5488'
    fit_color = '#E64B35'
    figdir = Path(figdir) / 'continuum'
    figdir.mkdir(parents=True, exist_ok=True)
    result = {}

    for k in data[ensembles[0]].keys():
        result[k] = {}
        for moment in sorted(data[ensembles[0]][k].keys()):
            result[k][moment] = {}
            tf_list = sorted(data[ensembles[0]][k][moment].keys())
            for tf in tf_list:
                x = []
                data_matrix = []
                err = []
                ensemble_list = []
                t_over_t0 = []
                for ensemble in ensembles:
                    if tf not in data[ensemble][k][moment]: continue
                    samples = data[ensemble][k][moment][tf]
                    x.append(params[ensemble]['spacing'] ** 2 / metadata[ensemble]['t0'])
                    data_matrix.append(samples)
                    err.append(tp.cal_err(samples,tech))
                    ensemble_list.append(ensemble)
                    flow_times = np.asarray(metadata[ensemble].get('flow_times',np.asarray(metadata[ensemble]['tau_list']) * metadata[ensemble]['flow_dt']))
                    t_over_t0.append(flow_times[tf] / metadata[ensemble]['t0'])

                selected = fit_range[k][moment]
                fit_index = np.asarray([i for i, ensemble in enumerate(ensemble_list) if ensemble in selected],dtype=int)
                data_matrix = np.column_stack(data_matrix)
                result_para = fitting(np.asarray(x)[fit_index],data_matrix[:,fit_index])
                result[k][moment][tf] = {'t_over_t0': np.mean(t_over_t0), 't_over_t0_values': np.asarray(t_over_t0), 'ensembles': ensemble_list, 'a2_over_t0': np.asarray(x), 'samples': data_matrix, 'mean': tp.cal_mean(data_matrix), 'err': np.asarray(err), 'fit_indices': fit_index, 'fit': result_para}

            tf_list = sorted(result[k][moment].keys())
            ncols = 5
            nrows = int(np.ceil(len(tf_list) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
            axes = axes.reshape(-1)
            for axis, tf in zip(axes, tf_list):
                result_now = result[k][moment][tf]
                x = result_now['a2_over_t0']
                mean = result_now['mean']
                err = result_now['err']
                fit_index = result_now['fit_indices']
                data_fit_mask = np.zeros(len(x), dtype=bool)
                data_fit_mask[fit_index] = True
                axis.errorbar(x[~data_fit_mask],mean[~data_fit_mask],yerr=err[~data_fit_mask],ls='None',marker='o',color=data_color,mec=data_color,alpha=0.2,capsize=2,fillstyle='none')
                axis.errorbar(x[data_fit_mask],mean[data_fit_mask],yerr=err[data_fit_mask],ls='None',marker='o',color=data_color,mec=data_color,capsize=2,fillstyle='none')
                result_para = result_now['fit']
                fit_x = np.unique(np.concatenate((np.linspace(0,max(x)*1.15,100),[x[fit_index[0]],x[fit_index[-1]]])))
                recon_matrix = np.zeros([len(result_para['samples']),len(fit_x)])
                for ls in range(len(result_para['samples'])):
                    recon_matrix[ls] = fit_function(fit_x,result_para['samples'][ls])
                fit_mean = tp.cal_mean(recon_matrix)
                fit_err = tp.cal_err(recon_matrix,tech)
                fit_mask = (fit_x >= min(x[fit_index])) & (fit_x <= max(x[fit_index]))
                axis.fill_between(fit_x,fit_mean-fit_err,fit_mean+fit_err,color=fit_color,alpha=0.20,edgecolor='none')
                axis.fill_between(fit_x[fit_mask],fit_mean[fit_mask]-fit_err[fit_mask],fit_mean[fit_mask]+fit_err[fit_mask],color=fit_color,alpha=0.45,edgecolor='none')
                axis.errorbar([0],[result_para['mean'][0]],yerr=[result_para['err'][0]],marker='s',markersize=6,color='k',capsize=2)
                axis.set_title(r'$t_f/t_0=%.3g$' % result_now['t_over_t0'])
                axis.set_xlabel(r'$a^2/t_0$')
                axis.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (moment - 1))
                ymin = min(np.min(mean-err),np.min(fit_mean-fit_err),result_para['mean'][0]-result_para['err'][0])
                ymax = max(np.max(mean+err),np.max(fit_mean+fit_err),result_para['mean'][0]+result_para['err'][0])
                ypad = 0.05 * max(ymax - ymin, np.finfo(float).eps)
                axis.set_ylim([ymin - ypad, ymax + ypad])
                axis.tick_params(axis='both', direction='in', labelsize=8)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
            for axis in axes[len(tf_list):]:
                axis.axis('off')
            fig.suptitle(r'$%s\quad \mathrm{continuum}$' % inputs.labels(k))
            fig.tight_layout()
            fig.savefig(figdir / ('ratio_%s_moment%d.pdf' % (k,moment)),transparent=True)
            tp.show_in_spyder()
            plt.close(fig)
    return result


def flow_extrapolation(continuum, fit_range, figdir):
    tech = 'bootstrap'
    data_color = '#3C5488'
    fit_color = '#E64B35'
    figdir = Path(figdir) / 'flow'
    figdir.mkdir(parents=True, exist_ok=True)
    result = {}
    for k in continuum:
        result[k] = {}
        for moment in sorted(continuum[k]):
            tf_list = sorted(continuum[k][moment].keys())
            x = np.asarray([continuum[k][moment][tf]['t_over_t0'] for tf in tf_list])
            data_matrix = np.column_stack([continuum[k][moment][tf]['fit']['samples'][:,0] for tf in tf_list])
            mean = tp.cal_mean(data_matrix)
            err = tp.cal_err(data_matrix,tech)
            selected = fit_range[k][moment]
            fit_index = np.arange(max(0,selected[0]),min(selected[1],len(tf_list)-1)+1,dtype=int)
            result_para = fitting(x[fit_index],data_matrix[:,fit_index])
            result[k][moment] = {'tf': np.asarray(tf_list,dtype=int), 't_over_t0': x, 'samples': data_matrix, 'mean': mean, 'err': err, 'fit_indices': fit_index, 'fit': result_para}

            fig, ax = plt.subplots(figsize=(6.4, 4))
            ax.errorbar(x,mean,yerr=err,ls='None',marker='o',color=data_color,mec=data_color,capsize=2,fillstyle='none')
            fit_x = np.unique(np.concatenate((np.linspace(0,max(x),100),[x[fit_index[0]],x[fit_index[-1]]])))
            recon_matrix = np.zeros([len(result_para['samples']),len(fit_x)])
            for ls in range(len(result_para['samples'])):
                recon_matrix[ls] = fit_function(fit_x,result_para['samples'][ls])
            fit_mean = tp.cal_mean(recon_matrix)
            fit_err = tp.cal_err(recon_matrix,tech)
            fit_mask = (fit_x >= x[fit_index[0]]) & (fit_x <= x[fit_index[-1]])
            ax.fill_between(fit_x,fit_mean-fit_err,fit_mean+fit_err,color=fit_color,alpha=0.20,edgecolor='none')
            ax.fill_between(fit_x[fit_mask],fit_mean[fit_mask]-fit_err[fit_mask],fit_mean[fit_mask]+fit_err[fit_mask],color=fit_color,alpha=0.45,edgecolor='none')
            ax.errorbar([0],[result_para['mean'][0]],yerr=[result_para['err'][0]],marker='s',markersize=6,color='k',capsize=2)
            ax.set_xlabel(r'$t_f/t_0$')
            ax.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (moment - 1))
            ax.set_title(r'$%s\quad \mathrm{continuum}$' % inputs.labels(k))
            ymin = min(result_para['mean'][0]-result_para['err'][0],mean[-1]-err[-1])
            ymax = max(result_para['mean'][0]+result_para['err'][0],mean[-1]+err[-1])
            ypad = 0.05 * max(ymax - ymin, np.finfo(float).eps)
            ax.set_ylim([ymin - ypad, ymax + ypad])
            ax.tick_params(axis='both', direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(figdir / ('ratio_%s_moment%d.pdf' % (k,moment)),transparent=True)
            tp.show_in_spyder()
            plt.close(fig)

            print('%s moment=%d: %s' % (k,moment,repr(gv.gvar(result_para['mean'][0],result_para['err'][0]))))
    return result
