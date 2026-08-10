#!/public/home/liuchuan/software/miniconda3/bin/python3
# Author: Haobo Yan
#%%
import os
import argparse
from re import match
import sys
import numpy as np
import funcs as tp
from scipy import linalg
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import tqdm 
import gvar as gv
import warnings
import pickle
import inputs
import setup
import reading
import reading_etmc
import conf_tests
import preprocess
import preprocess_etmc
import plotdata
import plotdata_etmc
import fit_two
import fit_three
import build_R_pdf
import quantities
import ratio

#%%
# 2026.05.03: Fix bugs from copy and cross check.
# 2026.03.02: Start the PDF project. Copy from the Sigmac project.
# 2025.06.03: Start the Sigmac project. Copy from the Tcc project.

if __name__ == '__main__':
    #%%
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser(description='Do PDF analysis.')
        parser.add_argument('-e','--ensemble', type = str, required = True, metavar = '', help='ensemble')
        parser.add_argument('-tc','--tech', type = str, required = True, metavar = '', help='technology')
        parser.add_argument('-pd','--plotdata', type = int, required = True, metavar = '', help='plot the data')
        parser.add_argument('-two','--two', type = int, required = True, metavar = '', help='fit the single particles')
        parser.add_argument('-three','--three', type = int, default = 0, metavar = '', help='fit the three-point functions')
        parser.add_argument('-r2','--read2', type = str, required = True, metavar = '', help='read 2pt')
        parser.add_argument('-r3','--read3', type = str, required = True, metavar = '', help='read 3pt')
        args = parser.parse_args()
    else:
        class Args:
            ensemble = 'test'
            ensemble = 'cC211'
            tech = 'jackknife'
            plotdata = 0
            two = 0
            three = 0
            read2 = 'fast'
            read3 = 'fast'
        args = Args()
    print(args)

    #%%
    etmc = args.ensemble in inputs.ETMC_ENSEMBLES
    params = inputs.cal_params(args)
    datadir = params['datadir']

    setup.setup(params)
    warnings.filterwarnings("ignore")

    options = {}
    options['two'] = args.two
    options['three'] = args.three
    options['plotdata'] = args.plotdata
    options['conf_tests'] = 0

    print('N = %d'%(params['N']))
    print('Nb = %d'%(params['Nb']))


    #%%
    # Read the data
    if etmc:
        data2, data3, metadata = reading_etmc.reading(params, args.read2, args.read3)
    else:
        [data2, data3] = reading.reading(params, args.read2, args.read3)

        # For cross check
        if args.read3 != 'no':
            pdf_dir = '../%s/PDF' % datadir['mydata']
            os.makedirs(pdf_dir, exist_ok=True)
            for key in sorted(data3.keys()):
                if key.startswith(('pion-cov-nder_', 'pion-PDF-n_', 'kaon-cov-nder_', 'kaon-PDF-n_', 'kaon_s-cov-nder_', 'kaon_s-PDF-n_')):
                    np.save('%s/%s.npy' % (pdf_dir, key), data3[key])

    #%%
    # Conf test
    if options['conf_tests']:
        conf_tests.conf_dist(params, data2, data3)
        conf_tests.bin_test(params, data2)
        # conf_tests.boots_test(params, data2)

    #%%
    # Preprocess the data
    relist = tp.resamplelist(params['Nb'], params) # Need only the shape of the data
    relen = relist.shape[0]
    params['relen'] = relen

    if etmc:
        [data2, data3] = preprocess_etmc.preprocess(params, relist, data2, data3)
    else:
        [data2, data3] = preprocess.preprocess(args, datadir, params, relist, data2, data3)

    #%%
    # Plot the original data
    if options['plotdata']:
        if etmc:
            [xR, R] = build_R_pdf.build_R_pdf(params, data2, data3, metadata, result=None)
            plotdata_etmc.plotdata(params, data2, data3, metadata, two=True, three=True, three_pdf=True, xR=xR, R=R, result=None, label='plain')
        else:
            plotdata.plotdata(params, data2, data3, mtype='exp', two=True, three=True, three_pdf=True, label='plain')

    #%%
    # One particle fit
    if options['two']:
        params['just_changing_tmin'] = 1
        params['lazy_tmin'] = 0
        [fdata2, result_para, result_chi2dof, result, ans] = fit_two.fit_two(params, data2, mtype='cosh', obj='corr')

        with open('../%s/spectra/%s/results_two_%s_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],params['tech']),'wb') as dfile:
            pickle.dump([data2, fdata2, result_para, result_chi2dof, result, ans],dfile)

    else:
        with open('../%s/spectra/%s/results_two_%s_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],params['tech']),'rb') as dfile:
            [data2, fdata2, result_para, result_chi2dof, result, ans] = pickle.load(dfile)

    #%%
    # # Dispersion relation
    # if options['dispersion']:
    #     params['just_changing_tmin'] = 0
    #     [disfdata, disresult_para, disresult_chi2dof, disresult] = dispersion.dispersion(params, selected, result, sv=str(params['ens']))

    #     with open('../%s/spectra/%s/dispersion_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'wb') as dfile:
    #         pickle.dump([disfdata, disresult_para, disresult_chi2dof, disresult],dfile)

    # else:
    #     with open('../%s/spectra/%s/dispersion_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'rb') as dfile:
    #         [disfdata, disresult_para, disresult_chi2dof, disresult] = pickle.load(dfile)

    #%%
    # From here assume using ETMC data and do not write explicitly "if etmc" anymore.
    [xR, R] = build_R_pdf.build_R_pdf(params, data2, data3, metadata, result)
    # Plot the 3pt data again with fitted mass
    if options['plotdata']:
        plotdata_etmc.plotdata(params, data2, data3, metadata, two=False, three=True, three_pdf=False, xR=xR, R=R, result=result, label='mass')

    #%%
    # 3pt fit
    if options['three']:
        params['just_changing_tins'] = 0
        params['lazy_tins'] = 0
        result_chi2dof, result_3pt = fit_three.fit_three(params, xR, R, metadata, result)

        with open('../%s/spectra/%s/results_three_%s_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],params['tech']),'wb') as dfile:
            pickle.dump([result_chi2dof, result_3pt],dfile)
    else:
        with open('../%s/spectra/%s/results_three_%s_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],params['tech']),'rb') as dfile:
            [result_chi2dof, result_3pt] = pickle.load(dfile)

    #%%
    # Plot the largest-tsep result versus flow time as the final analysis step.
    largest_tsep_ratios = ratio.largest_tsep_result(result_3pt, params['tsep_list'])
    ratio.plot_ratio_vs_tf(largest_tsep_ratios, params['tech'], params['ensemble'], '../%s' % params['figures'])
    with open('../%s/spectra/%s/results_three_largest_tsep_%s.pckl' % (
            datadir['mydata'], params['ensemble'], params['tech']), 'wb') as dfile:
        pickle.dump(largest_tsep_ratios, dfile)

    #%%
    # Some physical quantities
    quantities.quantities(params, relen, result)
