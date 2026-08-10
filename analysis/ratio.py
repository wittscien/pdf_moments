"""Select and plot the fitted PDF-moment ratios."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import funcs as tp
import inputs


DATA_COLOR = '#3C5488'


def largest_tsep_result(result_3pt, tsep_list=None):
    """Use the largest available ``tsep`` for every ``(k, moment, tf)``."""
    largest_tsep_ratios = {}
    for k in result_3pt:
        largest_tsep_ratios[k] = {}
        for moment in result_3pt[k]:
            largest_tsep_ratios[k][moment] = {}
            for tf in result_3pt[k][moment]:
                available = result_3pt[k][moment][tf]
                tseps = list(available.keys())
                if tsep_list is not None:
                    preferred = [tsep for tsep in tsep_list if tsep in available]
                    if preferred:
                        tseps = preferred
                tsep = max(tseps, key=int)
                largest_tsep_ratios[k][moment][tf] = np.asarray(
                    [available[tsep][ls][0] for ls in sorted(available[tsep], key=int)],
                    dtype=float,
                )
    return largest_tsep_ratios


def plot_ratio_vs_tf(largest_tsep_ratios, tech, ensemble, figure_dir):
    """Plot the largest-``tsep`` ratio against the integer flow step."""
    figure_dir = Path(figure_dir) / 'ratio_tf' / ensemble
    figure_dir.mkdir(parents=True, exist_ok=True)

    for k in largest_tsep_ratios:
        for moment in sorted(largest_tsep_ratios[k]):
            tf_list = sorted(largest_tsep_ratios[k][moment], key=int)
            x = np.asarray(tf_list, dtype=float)
            y = np.asarray([tp.cal_mean(largest_tsep_ratios[k][moment][tf]) for tf in tf_list])
            yerr = np.asarray([tp.cal_err(largest_tsep_ratios[k][moment][tf], tech)[()] for tf in tf_list])

            fig, ax = plt.subplots(figsize=(6.4, 4))
            ax.errorbar(x, y, yerr=yerr, ls='None', marker='o', color=DATA_COLOR, mec=DATA_COLOR, capsize=2, fillstyle='none', label=r'$t_{\mathrm{sep}}=\mathrm{max}$')
            ax.set_xlabel(r'$t_f$')
            ax.set_ylabel(r'$\langle x^{%d}\rangle/\langle x\rangle$' % (moment - 1))
            ax.set_title(r'$\mathrm{Ens}=%s\quad %s$' % (ensemble, inputs.labels(k)))
            ax.set_ylim([min(y[3] - 3 * yerr[3], y[-1] - 3 * yerr[-1]), max(y[3] + 10 * yerr[3], y[-1] + 10 * yerr[-1])])
            ax.tick_params(axis='both', direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_dir / ('ratio_%s_moment%d_tf.pdf' % (k, moment)), transparent=True)
            tp.show_in_spyder()
            plt.close(fig)
