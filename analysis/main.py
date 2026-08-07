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
import quantities

# In[]
# 2026.05.03: Fix bugs from copy and cross check.
# 2026.03.02: Start the PDF project. Copy from the Sigmac project.
# 2025.06.03: Start the Sigmac project. Copy from the Tcc project.

if __name__ == '__main__':
    # In[]
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser(description='Do PDF analysis.')
        parser.add_argument('-e','--ensemble', type = str, required = True, metavar = '', help='ensemble')
        parser.add_argument('-tc','--tech', type = str, required = True, metavar = '', help='technology')
        parser.add_argument('-pd','--plotdata', type = int, required = True, metavar = '', help='plot the data')
        parser.add_argument('-two','--two', type = int, required = True, metavar = '', help='fit the single particles')
        parser.add_argument('-r2','--read2', type = str, required = True, metavar = '', help='read 2pt')
        parser.add_argument('-r3','--read3', type = str, required = True, metavar = '', help='read 3pt')
        args = parser.parse_args()
    else:
        class Args:
            ensemble = 'C24P29'
            ensemble = 'test'
            ensemble = 'cC211'
            tech = 'jackknife'
            plotdata = 1
            two = 0
            read2 = 'fast'
            read3 = 'fast'
        args = Args()
    print(args)

    # In[]
    etmc = args.ensemble in inputs.ETMC_ENSEMBLES
    params = inputs.cal_params(args)
    datadir = params['datadir']

    setup.setup(params)
    warnings.filterwarnings("ignore")

    options = {}
    options['two'] = args.two
    options['plotdata'] = args.plotdata
    options['conf_tests'] = 1

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
        # conf_tests.bin_test(params, data2)
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

    # In[]
    # Plot the original data
    if options['plotdata']:
        if etmc:
            plotdata_etmc.plotdata(params, data2, data3, metadata, two=True, three=True, three_pdf=True, label='plain')
        else:
            plotdata.plotdata(params, data2, data3, mtype='exp', two=True, three=True, three_pdf=True, label='plain')

    # In[]
    # One particle fit
    if options['two']:
        params['just_changing_tmin'] = 0
        params['lazy_tmin'] = 0
        [fdata2, result_para, result_chi2dof, result, ans, energy_non_labels] = fit_two.fit_two(params, data2, mtype='cosh', obj='corr')

        dfile = open('../%s/spectra/%s/results_one_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'wb')
        all_results = [data2, fdata2, result_para, result_chi2dof, result, ans, energy_non_labels]
        pickle.dump(all_results,dfile)
        dfile.close()

    else:
        dfile = open('../%s/spectra/%s/results_one_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'rb')
        [data2, fdata2, result_para, result_chi2dof, result, ans, energy_non_labels] = pickle.load(dfile)
        dfile.close()

#     # In[]
#     # Dispersion relation
#     if options['dispersion']:
#         params['just_changing_tmin'] = 0
#         [disfdata, disresult_para, disresult_chi2dof, disresult] = dispersion.dispersion(params, selected, result, sv=str(params['ens']))

#         dfile = open('../%s/spectra/%s/dispersion_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'wb')
#         all_results = [disfdata, disresult_para, disresult_chi2dof, disresult]
#         pickle.dump(all_results,dfile)
#         dfile.close()

#     else:
#         dfile = open('../%s/spectra/%s/dispersion_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['tech']),'rb')
#         [disfdata, disresult_para, disresult_chi2dof, disresult] = pickle.load(dfile)
#         dfile.close()

#     # In[]
#     # Save results to params
#     params_oneres = saveparams.saveparams(result, relen)
#     params2.update(params_oneres)

#     # In[]
#     # Two particle data
#     # Correlation matrix
#     #weight = corr_weight.corr_weight(params, result)
#     weight='no'
#     few = corr_few.corr_few(params)
#     corrmat = calcorrmat.calcorrmat(params, data4, symmetrize=False, few=few, weight=weight)
#     dfile = open('../%s/corr/%s/corrmat_%s_%s_%d%d%d_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['isospin'],params['P'][0],params['P'][1],params['P'][2],params['irrep'],params['tech']),'wb')
#     pickle.dump(corrmat,dfile)
#     dfile.close()

#     # In[]
#     # Pure GEVP
#     if options['GEVP']:
#         [t0, tref, tv] = setting_t0.setting_t0(params)
#         params2['t0'] = t0
#         params2['mtype'] = 'Gexp'
#         params2['just_changing_tmin'] = 0
#         params2['lazy_tmin'] = 0
#         only_plot = 0
#         [Gdata, Gfdata, Gresult_para, Gresult_chi2dof, Gresult, Gans] = fit_GEVP.fit_GEVP(params, params2, corrmat, t0, tref, tv, mtype=params2['mtype'], selected2=selected2, weighto=weight, only_plot=only_plot)

#         dfile = open('../%s/spectra/%s/results_two_%s_%s_%d%d%d_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['isospin'],params['P'][0],params['P'][1],params['P'][2],params['irrep'],params['tech']),'wb')
#         all_results = [Gdata, Gfdata, Gresult_para, Gresult_chi2dof, Gresult, Gans]
#         pickle.dump(all_results,dfile)
#         dfile.close()
#     else:
#         dfile = open('../%s/spectra/%s/results_two_%s_%s_%d%d%d_%s_%s.pckl'%(datadir['mydata'],params['ensname'],params['ensname'],params['isospin'],params['P'][0],params['P'][1],params['P'][2],params['irrep'],params['tech']),'rb')
#         [Gdata, Gfdata, Gresult_para, Gresult_chi2dof, Gresult, Gans] = pickle.load(dfile)
#         dfile.close()


#     # In[]
#     # Some physical quantities
#     if options['GEVP']:
#         quantities.quantities(params, params2, relen, result, Gresult, weight)


# # %%
