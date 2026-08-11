from pathlib import Path
import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import betaln
import funcs as tp


def moment_ratio(m, para):
    [alpha, beta] = para
    # For f_v(x)=N*x^alpha*(1-x)^beta, the normalization N cancels in <x^m>/<x>.
    return np.exp(betaln(alpha + np.asarray(m) + 1,beta + 1) - betaln(alpha + 2,beta + 1))


def pdf_function(x, para):
    [alpha, beta] = para
    # The valence-number sum rule integral_0^1 f_v(x) dx=1 fixes N=1/B(alpha+1,beta+1).
    return np.exp(-betaln(alpha + 1,beta + 1) + alpha * np.log(x) + beta * np.log1p(-x))


def chi(para, data, Linv, m, use_prior):
    [log_alpha1, beta] = para
    alpha = np.exp(log_alpha1) - 1
    data_row = np.dot(Linv,data - moment_ratio(m,[alpha,beta]))
    if not use_prior: return data_row
    # These are the priors used in arXiv:2510.26738: log(alpha+1)~N(-0.6,0.2), beta~N(0,5).
    prior_row = np.asarray([(log_alpha1 + 0.6) / 0.2,beta / 5])
    return np.concatenate((data_row,prior_row))


def fitting(data, use_prior):
    tech = 'bootstrap'
    moment_list = [3,4,5,6]
    # Operator moment n corresponds to the PDF power m=n-1; here n=3,...,6 gives m=2,...,5.
    m = np.asarray([moment - 1 for moment in moment_list])
    samples = np.column_stack([data[moment] for moment in moment_list])
    cov = tp.cal_cov(samples,tech)
    Linv = np.linalg.inv(np.linalg.cholesky(cov))

    # Fit the central sample first and use it as the common starting point for all joint bootstrap fits.
    # Finite lower bounds keep alpha and beta strictly above -1 even when a no-prior fit runs to the boundary.
    initial = np.asarray([-0.6,1.0])
    central_fit = least_squares(chi,initial,args=(samples[0],Linv,m,use_prior),bounds=([np.log(1e-8),-1 + 1e-8],[np.inf,np.inf]))
    para_matrix = np.zeros([len(samples),2])
    para_matrix[0] = [np.exp(central_fit.x[0]) - 1,central_fit.x[1]]
    for ls in range(1,len(samples)):
        fit = least_squares(chi,central_fit.x,args=(samples[ls],Linv,m,use_prior),bounds=([np.log(1e-8),-1 + 1e-8],[np.inf,np.inf]))
        para_matrix[ls] = [np.exp(fit.x[0]) - 1,fit.x[1]]

    # The reported chi2 contains only the moment-ratio residuals; the two prior residuals are not data points.
    chi2 = np.sum(central_fit.fun[:len(m)] ** 2)
    dof = len(m) - 2
    return m, samples, para_matrix, chi2, dof


def pdf_reconstruction(k, data, use_prior, figdir):
    tech = 'bootstrap'
    data_color = {'pion': '#C84C5A', 'kaon': '#D18F32', 'kaon_s': '#3D9970'}[k]
    band_color = {'pion': '#9DC5DF', 'kaon': '#EBCB8B', 'kaon_s': '#A8D5BA'}[k]
    pdf_label = {'pion': r'$x f_v^\pi(x)$', 'kaon': r'$x f_v^{K,u}(x)$', 'kaon_s': r'$x f_v^{K,s}(x)$'}[k]
    figdir = Path(figdir) / 'pdf'
    figdir.mkdir(parents=True, exist_ok=True)

    m, samples, para_matrix, chi2, dof = fitting(data,use_prior)
    result = {'alpha': para_matrix[:,0], 'beta': para_matrix[:,1]}

    # Reconstruct the fitted ratios and x*f_v(x) for every bootstrap sample before calculating plot errors.
    fit_m = np.linspace(1,max(m),200)
    ratio_matrix = np.asarray([moment_ratio(fit_m,para) for para in para_matrix])
    x = np.linspace(1e-4,1 - 1e-4,500)
    pdf_matrix = np.asarray([x * pdf_function(x,para) for para in para_matrix])
    ratio_mean = tp.cal_mean(samples)
    ratio_err = tp.cal_err(samples,tech)
    fit_mean = tp.cal_mean(ratio_matrix)
    fit_err = tp.cal_err(ratio_matrix,tech)
    pdf_mean = tp.cal_mean(pdf_matrix)
    pdf_err = tp.cal_err(pdf_matrix,tech)

    plot_style = {'font.family': 'serif', 'font.serif': ['STIXGeneral','DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 10, 'axes.linewidth': 0.8}
    with plt.rc_context(plot_style):
        fig, axes = plt.subplots(1,2,figsize=(9.2,3.7),gridspec_kw={'width_ratios': [1,1.12]})
        axes[0].axhline(0,color='0.78',lw=0.7,zorder=0)
        axes[0].fill_between(fit_m,fit_mean-fit_err,fit_mean+fit_err,color=band_color,alpha=0.55,edgecolor='none',label='fit')
        axes[0].errorbar(m,ratio_mean,yerr=ratio_err,ls='None',marker='o',markersize=5,color=data_color,mec=data_color,mfc='white',mew=1.2,capsize=2.5,elinewidth=1,label='lattice',zorder=3)
        axes[0].set_xlim([0.7,max(m) + 0.3])
        axes[0].set_xticks(np.arange(1,max(m) + 1))
        axes[0].set_xlabel(r'$m$')
        axes[0].set_ylabel(r'$\langle x^m\rangle/\langle x\rangle$')
        axes[0].legend(frameon=False,ncol=2,loc='upper center',handlelength=1.5,columnspacing=1.2)

        axes[1].fill_between(x,pdf_mean-pdf_err,pdf_mean+pdf_err,color=band_color,alpha=0.55,edgecolor='none',label=pdf_label)
        axes[1].set_xlim([0,1])
        axes[1].set_ylim([0,1.12 * np.max(pdf_mean + pdf_err)])
        axes[1].set_xlabel(r'$x$')
        axes[1].set_ylabel(r'$x f_v(x)$')
        axes[1].legend(frameon=False,loc='upper right',handlelength=1.5)
        for axis in axes:
            axis.minorticks_on()
            axis.tick_params(axis='both',which='both',direction='in')
            axis.tick_params(which='major',length=4,width=0.8)
            axis.tick_params(which='minor',length=2,width=0.6)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
        fig.tight_layout(w_pad=2)
        fig.savefig(figdir / ('%s_pdf.pdf' % k),transparent=True,bbox_inches='tight')
        tp.show_in_spyder()
        plt.close(fig)

    alpha = gv.gvar(tp.cal_mean(result['alpha']),tp.cal_err(result['alpha'],tech))
    beta = gv.gvar(tp.cal_mean(result['beta']),tp.cal_err(result['beta'],tech))
    print('%s PDF: prior=%s, alpha=%s, beta=%s, chi2/dof=%.2f/%d' % (k,use_prior,repr(alpha),repr(beta),chi2,dof))
    return result
