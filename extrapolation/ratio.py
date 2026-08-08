"""Utilities for selecting and plotting the fitted PDF-moment ratios.

The three-point fit result is expected to have the same resampling layout as
the two-point ``result`` object in ``analysis/funcs.py``::

    result[k][n][tf][tsep][ls] = constant fit result for resample ``ls``

The first resample is the central value and the remaining resamples are used
by ``analysis.funcs.cal_err`` for jackknife/bootstrap errors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import funcs as tp
import inputs

DATA_COLOR = '#3C5488'

def largest_tsep_result(result_3pt, tsep_list=None):
    """Use the largest available ``tsep`` for every ``(k, n, tf)``.

    Parameters
    ----------
    result_3pt:
        Nested fitted results ``result[k][n][tf][tsep][ls]``.
    tsep_list:
        Optional preferred list.  The actual maximum key present in each fit
        is used, so incomplete ``tsep`` output remains visible rather than
        being silently padded.

    Returns
    -------
    selected, selected_tsep:
        ``selected[k][n][tf]`` is a resample array.  ``selected_tsep`` records
        which source-sink separation was selected for each entry.
    """
    selected = {}
    selected_tsep = {}
    for k in result_3pt:
        selected[k] = {}
        selected_tsep[k] = {}
        for n in result_3pt[k]:
            selected[k][n] = {}
            selected_tsep[k][n] = {}
            for tf in result_3pt[k][n]:
                available = result_3pt[k][n][tf]
                tseps = list(available.keys())
                if tsep_list is not None:
                    preferred = [tsep for tsep in tsep_list if tsep in available]
                    if preferred:
                        tseps = preferred
                tsep = max(tseps, key=int)
                selected[k][n][tf] = np.asarray([available[tsep][ls][0] for ls in sorted(available[tsep], key=int)], dtype=float)
                selected_tsep[k][n][tf] = int(tsep)
    return selected, selected_tsep


def plot_ratio_vs_tf(selected, metadata, tech, ensemble, figure_dir, xaxis='tf'):
    """Plot each moment ratio selected at the largest ``tsep`` versus flow.

    ``xaxis='tf'`` gives the integer flow-step plot requested for the first
    check.  Passing ``xaxis='t/t0'`` uses the dimensionless flow time while
    keeping the same selected resampling results.
    """
    figure_dir = Path(figure_dir) / 'ratio_tf' / ensemble
    figure_dir.mkdir(parents=True, exist_ok=True)

    tau_list = np.asarray(metadata['tau_list'], dtype=float)
    flow_times = np.asarray(metadata.get('flow_times', tau_list * metadata['flow_dt']), dtype=float)
    flow_ratio = flow_times / float(metadata['t0'])

    for k in selected:
        for n in sorted(selected[k]):
            tf_list = sorted(selected[k][n], key=int)
            x = np.asarray(tf_list, dtype=float)
            if xaxis == 't/t0':
                x = flow_ratio[tf_list]
            y = np.asarray([tp.cal_mean(selected[k][n][tf]) for tf in tf_list])
            yerr = np.asarray([tp.cal_err(selected[k][n][tf], tech)[()] for tf in tf_list])

            fig, ax = plt.subplots(figsize=(6.4, 4))
            ax.errorbar(x, y, yerr=yerr, ls='None', marker='o', color=DATA_COLOR, mec=DATA_COLOR, capsize=2, fillstyle='none', label=r'$t_{\mathrm{sep}}=\mathrm{max}$')
            ax.set_xlabel(r'$t_f$' if xaxis == 'tf' else r'$t_f/t_0$')
            ax.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (n - 1))
            ax.set_title(r'$\mathrm{Ens}=%s\quad %s$' % (ensemble, inputs.labels(k)))
            ax.set_ylim([min(y[3] - 3 * yerr[3], y[-1] - 3 * yerr[-1]), max(y[3] + 10 * yerr[3], y[-1] + 10 * yerr[-1])])
            ax.tick_params(axis='both', direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
            fig.tight_layout()
            suffix = 'tf' if xaxis == 'tf' else 't_over_t0'
            fig.savefig(figure_dir / ('ratio_%s_n%d_%s.pdf' % (k, n, suffix)), transparent=True)
            tp.show_in_spyder()
            plt.close(fig)
