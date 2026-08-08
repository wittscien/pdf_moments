"""Continuum and zero-flow extrapolations for fitted PDF-moment ratios."""

from pathlib import Path

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import funcs as tp
import inputs

DATA_COLOR = '#3C5488'
FIT_COLOR = '#E64B35'

def linear_fit(x, y, yerr):
    """Fit ``y = intercept + slope*x`` with the analysis-style least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    sigma = np.where(yerr > 0, yerr, np.finfo(float).eps)

    prior = [np.average(y, weights=1 / sigma ** 2), 0.0]
    fit = least_squares(lambda parameter: (y - parameter[0] - parameter[1] * x) / sigma, prior)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        'mean': fit.x,
        'err': error,
        'cov': covariance,
        'chi2': float(np.sum(fit.fun ** 2)),
        'dof': int(len(y) - len(prior)),
    }


def continuum_extrapolation(selected_by_ensemble, ensembles, params_by_ensemble, metadata_by_ensemble, tech, fit_ranges, figure_dir):
    """At each flow time, extrapolate the ratio linearly in ``a^2/t0``.

    The input ratios are resample arrays from ``ratio.largest_tsep_result``.
    Their central values and errors are calculated with ``analysis.funcs``;
    the continuum fit then uses those errors as independent ensemble errors.
    """
    figure_dir = Path(figure_dir) / 'continuum'
    figure_dir.mkdir(parents=True, exist_ok=True)
    continuum = {}

    for k in selected_by_ensemble[ensembles[0]]:
        continuum[k] = {}
        for n in sorted(selected_by_ensemble[ensembles[0]][k]):
            records = {}
            for tf in sorted({tf for ensemble in ensembles if k in selected_by_ensemble[ensemble] and n in selected_by_ensemble[ensemble][k] for tf in selected_by_ensemble[ensemble][k][n]}, key=int):
                x = []
                y = []
                yerr = []
                used_ensembles = []
                flow_values = []
                for ensemble in ensembles:
                    if k not in selected_by_ensemble[ensemble] or n not in selected_by_ensemble[ensemble][k] or tf not in selected_by_ensemble[ensemble][k][n]:
                        continue
                    samples = selected_by_ensemble[ensemble][k][n][tf]
                    mean = float(tp.cal_mean(samples))
                    err = float(tp.cal_err(samples, tech))
                    metadata = metadata_by_ensemble[ensemble]
                    x.append(params_by_ensemble[ensemble]['spacing'] ** 2 / float(metadata['t0']))
                    y.append(mean)
                    yerr.append(err)
                    used_ensembles.append(ensemble)
                    tau = np.asarray(metadata['tau_list'], dtype=float)
                    flow_times = np.asarray(metadata.get('flow_times', tau * metadata['flow_dt']), dtype=float)
                    flow_values.append(flow_times[tf] / float(metadata['t0']))

                if len(x) < 2:
                    continue
                fit_ensembles = fit_ranges[k][n]
                fit_idx = np.asarray([i for i, ensemble in enumerate(used_ensembles) if ensemble in fit_ensembles], dtype=int)
                fit = linear_fit(np.asarray(x)[fit_idx], np.asarray(y)[fit_idx], np.asarray(yerr)[fit_idx])
                records[tf] = {
                    't_over_t0': float(np.mean(flow_values)),
                    't_over_t0_values': np.asarray(flow_values),
                    'ensembles': used_ensembles,
                    'a2_over_t0': np.asarray(x),
                    'mean': np.asarray(y),
                    'err': np.asarray(yerr),
                    'fit_indices': fit_idx,
                    'fit': fit,
                }

            continuum[k][n] = records
            tf_list = sorted(records, key=int)
            ncols = 5
            nrows = int(np.ceil(len(tf_list) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
            axes = axes.reshape(-1)
            for axis, tf in zip(axes, tf_list):
                record = records[tf]
                x = record['a2_over_t0']
                y = record['mean']
                yerr = record['err']
                fit_idx = record['fit_indices']
                data_fit_mask = np.zeros(len(x), dtype=bool)
                data_fit_mask[fit_idx] = True
                axis.errorbar(x[~data_fit_mask], y[~data_fit_mask], yerr=yerr[~data_fit_mask], ls='None', marker='o', color=DATA_COLOR, mec=DATA_COLOR, alpha=0.2, capsize=2, fillstyle='none')
                axis.errorbar(x[data_fit_mask], y[data_fit_mask], yerr=yerr[data_fit_mask], ls='None', marker='o', color=DATA_COLOR, mec=DATA_COLOR, capsize=2, fillstyle='none')
                fit = record['fit']
                fit_x = np.unique(np.concatenate((np.linspace(0, max(x) * 1.15, 100), [x[fit_idx[0]], x[fit_idx[-1]]])))
                fit_y = fit['mean'][0] + fit['mean'][1] * fit_x
                fit_err = np.sqrt(np.maximum(fit['cov'][0, 0] + 2 * fit_x * fit['cov'][0, 1] + fit_x ** 2 * fit['cov'][1, 1], 0))
                fit_mask = (fit_x >= min(x[fit_idx])) & (fit_x <= max(x[fit_idx]))
                axis.fill_between(fit_x, fit_y - fit_err, fit_y + fit_err, color=FIT_COLOR, alpha=0.20, edgecolor='none')
                axis.fill_between(fit_x[fit_mask], fit_y[fit_mask] - fit_err[fit_mask], fit_y[fit_mask] + fit_err[fit_mask], color=FIT_COLOR, alpha=0.45, edgecolor='none')
                axis.errorbar([0], [fit['mean'][0]], yerr=[fit['err'][0]], marker='s', markersize=6, color='k', capsize=2)
                axis.set_title(r'$t_f/t_0=%.3g$' % record['t_over_t0'])
                axis.set_xlabel(r'$a^2/t_0$')
                axis.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (n - 1))
                ymin = min(np.min(y - yerr), np.min(fit_y - fit_err), fit['mean'][0] - fit['err'][0])
                ymax = max(np.max(y + yerr), np.max(fit_y + fit_err), fit['mean'][0] + fit['err'][0])
                ypad = 0.05 * max(ymax - ymin, np.finfo(float).eps)
                axis.set_ylim([ymin - ypad, ymax + ypad])
                axis.tick_params(axis='both', direction='in', labelsize=8)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
            for axis in axes[len(tf_list):]:
                axis.axis('off')
            fig.suptitle(r'$%s\quad \mathrm{continuum}$' % inputs.labels(k))
            fig.tight_layout()
            fig.savefig(figure_dir / ('ratio_%s_n%d.pdf' % (k, n)), transparent=True)
            tp.show_in_spyder()
            plt.close(fig)
    return continuum


def flow_extrapolation(continuum, tech, fit_ranges, figure_dir):
    """Extrapolate the continuum values linearly to ``t/t0 = 0``.

    The intercept is printed for every hadron and moment and is returned in
    ``limits[k][n]['fit']`` for comparisons with experiment.
    """
    figure_dir = Path(figure_dir) / 'flow'
    figure_dir.mkdir(parents=True, exist_ok=True)
    limits = {}
    for k in continuum:
        limits[k] = {}
        for n in sorted(continuum[k]):
            records = continuum[k][n]
            tf_list = sorted(records, key=int)
            x = np.asarray([records[tf]['t_over_t0'] for tf in tf_list])
            y = np.asarray([records[tf]['fit']['mean'][0] for tf in tf_list])
            yerr = np.asarray([records[tf]['fit']['err'][0] for tf in tf_list])
            fit_range = fit_ranges[k][n]
            fit_idx = np.arange(max(0, fit_range[0]), min(fit_range[1], len(tf_list) - 1) + 1, dtype=int)
            fit = linear_fit(x[fit_idx], y[fit_idx], yerr[fit_idx])
            limits[k][n] = {
                'tf': np.asarray(tf_list, dtype=int),
                't_over_t0': x,
                'mean': y,
                'err': yerr,
                'fit_indices': fit_idx,
                'fit': fit,
            }

            fig, ax = plt.subplots(figsize=(6.4, 4))
            ax.errorbar(x, y, yerr=yerr, ls='None', marker='o', color=DATA_COLOR, mec=DATA_COLOR, capsize=2, fillstyle='none')
            fit_x = np.unique(np.concatenate((np.linspace(0, max(x), 100), [x[fit_idx[0]], x[fit_idx[-1]]])))
            fit_y = fit['mean'][0] + fit['mean'][1] * fit_x
            fit_err = np.sqrt(np.maximum(fit['cov'][0, 0] + 2 * fit_x * fit['cov'][0, 1] + fit_x ** 2 * fit['cov'][1, 1], 0))
            fit_mask = (fit_x >= x[fit_idx[0]]) & (fit_x <= x[fit_idx[-1]])
            ax.fill_between(fit_x, fit_y - fit_err, fit_y + fit_err, color=FIT_COLOR, alpha=0.20, edgecolor='none')
            ax.fill_between(fit_x[fit_mask], fit_y[fit_mask] - fit_err[fit_mask], fit_y[fit_mask] + fit_err[fit_mask], color=FIT_COLOR, alpha=0.45, edgecolor='none')
            ax.errorbar([0], [fit['mean'][0]], yerr=[fit['err'][0]], marker='s', markersize=6, color='k', capsize=2)
            ax.set_xlabel(r'$t_f/t_0$')
            ax.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (n - 1))
            ax.set_title(r'$%s\quad \mathrm{continuum}$' % inputs.labels(k))
            ymin = min(fit['mean'][0] - fit['err'][0], y[-1] - yerr[-1])
            ymax = max(fit['mean'][0] + fit['err'][0], y[-1] + yerr[-1])
            ypad = 0.05 * max(ymax - ymin, np.finfo(float).eps)
            ax.set_ylim([ymin - ypad, ymax + ypad])
            ax.tick_params(axis='both', direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(figure_dir / ('ratio_%s_n%d.pdf' % (k, n)), transparent=True)
            tp.show_in_spyder()
            plt.close(fig)

            print('%s n=%d: %s' % (k, n, repr(gv.gvar(fit['mean'][0], fit['err'][0]))))
    return limits
