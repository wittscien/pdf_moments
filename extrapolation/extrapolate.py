from pathlib import Path
import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import funcs as tp
import inputs
from flow_matching import c_numeric


def fit_function(x, para):
    [a, b] = para
    return a + b * x


def chi(para, data, Linv, x):
    chi_now = np.array([])
    for k in data.keys():
        row = data[k] - fit_function(x,para)
        chi_now = np.concatenate((chi_now,np.dot(Linv[k],row)))
    return chi_now


def fitting(x, samples, correlated):
    tech = 'bootstrap'
    x = np.asarray(x, dtype=float)
    samples = np.asarray(samples, dtype=float)
    # Use the straight-line fit to the central sample as the common bootstrap initial value.
    prior = np.polyfit(x,samples[0],1)[::-1]
    cov = tp.cal_cov(samples,tech)
    if not correlated:
        cov_diag = np.diag(cov)
        cov_diag = np.where(cov_diag == 0,1e-32,cov_diag)
        cov = np.diag(cov_diag)
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


def continuum_extrapolation(params, metadata, data, ensembles, fit_range, correlated, figdir):
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
            plot_result = {}
            tf_list = sorted(data[ensembles[0]][k][moment].keys())
            for tf in tf_list:
                # metadata['t0'] is sqrt(t0) in fm, while flow_times stores t/sqrt(t0) in fm.
                t_over_t0 = metadata[ensembles[0]]['flow_times'][tf] / metadata[ensembles[0]]['t0']
                x = []
                data_matrix = []
                ensemble_list = []
                for ensemble in ensembles:
                    if tf not in data[ensemble][k][moment]: continue
                    samples = data[ensemble][k][moment][tf]
                    x.append((params[ensemble]['spacing'] / metadata[ensemble]['t0']) ** 2)
                    data_matrix.append(samples)
                    ensemble_list.append(ensemble)

                selected = fit_range[k][moment]
                fit_index = np.asarray([i for i, ensemble in enumerate(ensemble_list) if ensemble in selected],dtype=int)
                data_matrix = np.column_stack(data_matrix)
                result_para = fitting(np.asarray(x)[fit_index],data_matrix[:,fit_index],correlated)
                result[k][moment][tf] = np.asarray([fit_function(0,para) for para in result_para['samples']])
                plot_result[tf] = {'t_over_t0': t_over_t0, 'a2_over_t0': np.asarray(x), 'samples': data_matrix, 'fit_indices': fit_index, 'fit': result_para}

            # Plot the continuum extrapolations.
            tf_list = sorted(result[k][moment].keys())
            ncols = 5
            nrows = int(np.ceil(len(tf_list) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
            axes = axes.reshape(-1)
            for axis, tf in zip(axes, tf_list):
                result_now = plot_result[tf]
                x = result_now['a2_over_t0']
                mean = tp.cal_mean(result_now['samples'])
                err = tp.cal_err(result_now['samples'],tech)
                fit_index = result_now['fit_indices']
                data_fit_mask = np.zeros(len(x), dtype=bool)
                data_fit_mask[fit_index] = True
                axis.errorbar(x[~data_fit_mask],mean[~data_fit_mask],yerr=err[~data_fit_mask],ls='None',marker='o',color=data_color,mec=data_color,alpha=0.2,capsize=2,fillstyle='none')
                axis.errorbar(x[data_fit_mask],mean[data_fit_mask],yerr=err[data_fit_mask],ls='None',marker='o',color=data_color,mec=data_color,capsize=2,fillstyle='none')
                result_para = result_now['fit']
                continuum_mean = tp.cal_mean(result[k][moment][tf])
                continuum_err = tp.cal_err(result[k][moment][tf],tech)
                fit_x = np.unique(np.concatenate((np.linspace(0,max(x)*1.15,100),[x[fit_index[0]],x[fit_index[-1]]])))
                recon_matrix = np.zeros([len(result_para['samples']),len(fit_x)])
                for ls in range(len(result_para['samples'])):
                    recon_matrix[ls] = fit_function(fit_x,result_para['samples'][ls])
                fit_mean = tp.cal_mean(recon_matrix)
                fit_err = tp.cal_err(recon_matrix,tech)
                fit_mask = (fit_x >= min(x[fit_index])) & (fit_x <= max(x[fit_index]))
                axis.fill_between(fit_x,fit_mean-fit_err,fit_mean+fit_err,color=fit_color,alpha=0.20,edgecolor='none')
                axis.fill_between(fit_x[fit_mask],fit_mean[fit_mask]-fit_err[fit_mask],fit_mean[fit_mask]+fit_err[fit_mask],color=fit_color,alpha=0.45,edgecolor='none')
                axis.errorbar([0],[continuum_mean],yerr=[continuum_err],marker='s',markersize=6,color='k',capsize=2)
                axis.set_title(r'$t_f/t_0=%.3g$' % result_now['t_over_t0'])
                axis.set_xlabel(r'$a^2/t_0$')
                axis.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (moment - 1))
                ymin = min(np.min(mean-err),np.min(fit_mean-fit_err),continuum_mean-continuum_err)
                ymax = max(np.max(mean+err),np.max(fit_mean+fit_err),continuum_mean+continuum_err)
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


def flow_extrapolation(params, metadata, data, continuum, ensembles, fit_range, correlated, figdir):
    tech = 'bootstrap'
    # Colors and markers for the finite-a data, continuum limit, and flow fit
    ensemble_color = {'cA211': '#4477AA', 'cB211': '#228833', 'cC211': '#CC6677'}
    ensemble_marker = {'cA211': 'o', 'cB211': 's', 'cC211': '^'}
    continuum_color = '#332288'
    fit_color = '#EE7733'
    figdir = Path(figdir) / 'flow'
    figdir.mkdir(parents=True, exist_ok=True)
    result = {}
    for k in continuum:
        result[k] = {}
        for moment in sorted(continuum[k]):
            flow_times = np.asarray(metadata[ensembles[0]]['flow_times'])
            tf_list = [tf for tf in sorted(continuum[k][moment].keys()) if flow_times[tf] > 0]
            x = flow_times[tf_list] / metadata[ensembles[0]]['t0']
            # Apply matching to the continuum-limit samples before the flow-time fit.
            sqrt_t0_GeVinv = metadata[ensembles[0]]['t0'] / params[ensembles[0]]['spacing'] * 1000 / params[ensembles[0]]['hca']
            flow_times_GeV2 = x * sqrt_t0_GeVinv ** 2
            matching_factor = np.asarray([c_numeric(2,t,2) / c_numeric(moment,t,2) for t in flow_times_GeV2])
            data_matrix = np.column_stack([continuum[k][moment][tf] for tf in tf_list]) * matching_factor
            selected = fit_range[k][moment]
            fit_index = np.asarray([i for i, tf in enumerate(tf_list) if selected[0] <= tf <= selected[1]],dtype=int)
            result_para = fitting(x[fit_index],data_matrix[:,fit_index],correlated)
            result[k][moment] = np.asarray([fit_function(0,para) for para in result_para['samples']])

            # Plot the flow-time extrapolation.
            mean = tp.cal_mean(data_matrix)
            err = tp.cal_err(data_matrix,tech)
            limit_mean = tp.cal_mean(result[k][moment])
            limit_err = tp.cal_err(result[k][moment],tech)

            fig, ax = plt.subplots(figsize=(6.4, 4))
            data_fit_mask = np.zeros(len(tf_list),dtype=bool)
            data_fit_mask[fit_index] = True
            ensemble_mean = []
            ensemble_err = []
            # Plot the data before continuum extrapolation; fade points outside the flow fit range
            for ensemble in ensembles:
                ensemble_x = np.asarray(metadata[ensemble]['flow_times'])[tf_list] / metadata[ensemble]['t0']
                ensemble_matrix = np.column_stack([data[ensemble][k][moment][tf] for tf in tf_list]) * matching_factor
                ensemble_y = tp.cal_mean(ensemble_matrix)
                ensemble_yerr = tp.cal_err(ensemble_matrix,tech)
                ensemble_mean.append(ensemble_y)
                ensemble_err.append(ensemble_yerr)
                ax.errorbar(ensemble_x[~data_fit_mask],ensemble_y[~data_fit_mask],yerr=ensemble_yerr[~data_fit_mask],ls='None',marker=ensemble_marker[ensemble],markersize=4,color=ensemble_color[ensemble],mec=ensemble_color[ensemble],capsize=2,fillstyle='none',alpha=0.12,zorder=1)
                ax.errorbar(ensemble_x[data_fit_mask],ensemble_y[data_fit_mask],yerr=ensemble_yerr[data_fit_mask],ls='None',marker=ensemble_marker[ensemble],markersize=4,color=ensemble_color[ensemble],mec=ensemble_color[ensemble],capsize=2,fillstyle='none',alpha=0.75,label=ensemble,zorder=2)
            ensemble_mean = np.asarray(ensemble_mean)
            ensemble_err = np.asarray(ensemble_err)
            # Plot the continuum-extrapolated data used for the flow-time extrapolation
            ax.errorbar(x[~data_fit_mask],mean[~data_fit_mask],yerr=err[~data_fit_mask],ls='None',marker='D',markersize=4.5,color=continuum_color,mec=continuum_color,capsize=2,fillstyle='none',alpha=0.18,zorder=2)
            ax.errorbar(x[data_fit_mask],mean[data_fit_mask],yerr=err[data_fit_mask],ls='None',marker='D',markersize=4.5,color=continuum_color,mec=continuum_color,capsize=2,fillstyle='none',label=r'$a^2\to 0$',zorder=3)
            fit_x = np.unique(np.concatenate((np.linspace(0,max(x),100),[x[fit_index[0]],x[fit_index[-1]]])))
            recon_matrix = np.zeros([len(result_para['samples']),len(fit_x)])
            for ls in range(len(result_para['samples'])):
                recon_matrix[ls] = fit_function(fit_x,result_para['samples'][ls])
            fit_mean = tp.cal_mean(recon_matrix)
            fit_err = tp.cal_err(recon_matrix,tech)
            fit_mask = (fit_x >= x[fit_index[0]]) & (fit_x <= x[fit_index[-1]])
            ax.fill_between(fit_x,fit_mean-fit_err,fit_mean+fit_err,color=fit_color,alpha=0.20,edgecolor='none')
            ax.fill_between(fit_x[fit_mask],fit_mean[fit_mask]-fit_err[fit_mask],fit_mean[fit_mask]+fit_err[fit_mask],color=fit_color,alpha=0.45,edgecolor='none',label='flow fit')
            ax.errorbar([0],[limit_mean],yerr=[limit_err],marker='P',markersize=7,color='k',capsize=2,label=r'$t_f\to 0$',zorder=4)
            ax.set_xlabel(r'$t_f/t_0$')
            ax.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (moment - 1))
            ax.set_title(r'$%s\quad \mathrm{continuum}$' % inputs.labels(k))
            # Set the y range from the fit region so noisy unused points do not compress the plot
            ymin = min(np.min(ensemble_mean[:,fit_index]-ensemble_err[:,fit_index]),np.min(mean[fit_index]-err[fit_index]),np.min(fit_mean-fit_err),limit_mean-limit_err)
            ymax = max(np.max(ensemble_mean[:,fit_index]+ensemble_err[:,fit_index]),np.max(mean[fit_index]+err[fit_index]),np.max(fit_mean+fit_err),limit_mean+limit_err)
            ypad = 0.05 * max(ymax - ymin, np.finfo(float).eps)
            ax.set_ylim([ymin - ypad, ymax + ypad])
            ax.tick_params(axis='both', direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False,ncol=2,fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / ('ratio_%s_moment%d.pdf' % (k,moment)),transparent=True)
            tp.show_in_spyder()
            plt.close(fig)

            print('%s moment=%d: %s' % (k,moment,repr(gv.gvar(limit_mean,limit_err))))
    return result
