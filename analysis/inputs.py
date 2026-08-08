from pathlib import Path
import pickle

import h5py as h5
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# ETMC data are stored in data/traceless_operators/.
ETMC_ENSEMBLES = ['cA211', 'cB211', 'cC211']


def cal_params(args):

    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True

    params = {}
    params['tech'] = args.tech
    params['ensemble'] = args.ensemble
    etmc = params['ensemble'] in ETMC_ENSEMBLES
    read2 = args.read2
    read3 = args.read3

    params['mommax'] = 1
    params['ns_min'] = 1
    params['ns_max'] = 2

    params['tau'] = 1

    params['srcs'] = {'sp'}

    params['datadir'] = {}
    params['datadir']['mydata'] = 'mydata/main'
    params['datadir']['2pt'] = 'data'
    params['datadir']['3pt'] = 'data'
    params['datadir']['4pt'] = 'data'
    params['figures'] = 'figures/main'

    params['key_2pt'] = ['pion', 'kaon']
    params['key_3pt'] = ['pion', 'kaon', 'kaon_s']

    if params['ensemble'] == "test":
        params['L'] = 4
        params['T'] = 8
        params['spacing'] = 0.2
        params['nsrc'] = 1
        params['nder'] = 3
        params['nflow'] = 1
        params['tsep_max_3pt'] = {'pion': 8, 'kaon': 8, 'kaon_s': 8}
        params['Z_V^l'] = 6
        params['confs'] = np.arange(1000,2020,20)
        params['confs'] = np.arange(1000,1080,20)
        params['confs'] = [0]

    elif params['ensemble'] == "C24P29":
        params['L'] = 24
        params['T'] = 72
        params['spacing'] = 0.10530
        params['nsrc'] = 2
        params['tsep_max_3pt'] = {'Sigmac': 23, 'pion': 72, 'kaon': 72, 'kaon_s': 72}
        params['tsep_max_4pt'] = {'Sigmac': 28}
        params['dt_list'] = {'Sigmac': [(1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,3), (3,4), (4,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9)]}
        max_dt = max(max(i, j) for (i, j) in params['dt_list']['Sigmac'])
        params['tJJ_max'] = {'Sigmac': params['tsep_max_4pt']['Sigmac'] - 2 * max_dt}
        params['Z_V^l'] = 0.79676
        params['Z_V^c'] = 1.57353
        exceptional = np.array([])
        params['confs'] = np.setdiff1d(np.arange(4050, 26550, 50),exceptional)

    elif params['ensemble'] == "C48P14":
        params['L'] = 48
        params['T'] = 96
        params['nsrc'] = 1
        params['spacing'] = 0.10530
        exceptional = np.array([])
        params['confs'] = np.setdiff1d(np.arange(2000,4640,20),exceptional)

    elif params['ensemble'] == "F32P30":
        params['L'] = 32
        params['T'] = 96
        params['nsrc'] = 1
        params['spacing'] = 0.07746
        exceptional = np.arange(10000,13000,50)
        params['confs'] = np.setdiff1d(np.arange(1000,20000,50),exceptional)
        params['confs'] = np.setdiff1d(np.arange(1000,10000,50),exceptional)

    elif params['ensemble'] == "F48P30":
        params['L'] = 48
        params['T'] = 96
        params['nsrc'] = 1
        params['spacing'] = 0.07746
        exceptional = np.array([])
        params['confs'] = np.setdiff1d(np.arange(6700,9880,20),exceptional)

    elif params['ensemble'] == "F32P21":
        params['L'] = 32
        params['T'] = 64
        params['nsrc'] = 1
        params['spacing'] = 0.07746
        exceptional = np.array([])
        params['confs'] = np.setdiff1d(np.arange(13500,36450,50),exceptional)

    elif params['ensemble'] == "F48P21":
        params['L'] = 48
        params['T'] = 96
        params['nsrc'] = 1
        params['spacing'] = 0.07746
        exceptional = np.array([5760,5780])
        params['confs'] = np.setdiff1d(np.arange(1620,6080,20),exceptional)

    elif params['ensemble'] == "H48P32":
        params['tau'] = 2
        params['L'] = 48
        params['T'] = 144
        params['nsrc'] = 1
        params['spacing'] = 0.05187
        exceptional = np.array([])
        params['confs'] = np.setdiff1d(np.arange(1000,6480,20),exceptional)

    elif params['ensemble'] == 'cA211':
        params['L'] = 32
        params['T'] = 64
        params['spacing'] = 0.0922
        params['Z_V^l'] = 1
        params['Z_V^s'] = 1
        params['tsep_list'] = [24, 28, 32]
        exceptional = []

    elif params['ensemble'] == 'cB211':
        params['L'] = 48
        params['T'] = 96
        params['spacing'] = 0.0800
        params['Z_V^l'] = 1
        params['Z_V^s'] = 1
        params['tsep_list'] = [28, 32, 36]
        # Incomplete kaon_s flow data at dt=36.
        exceptional = [
            ('cB211b.25.48', 544),
            ('cB211b.25.48', 568),
        ]

    elif params['ensemble'] == 'cC211':
        params['L'] = 48
        params['T'] = 96
        params['spacing'] = 0.0684
        params['Z_V^l'] = 0.73
        params['Z_V^s'] = 0.73
        params['tsep_list'] = [32, 40, 48]
        exceptional = []

    elif params['ensemble'] == 'cD211':
        params['L'] = 96
        params['T'] = 192
        params['spacing'] = 0.0573
        params['Z_V^l'] = 1
        params['Z_V^s'] = 1
        params['tsep_list'] = [32, 40, 48]
        exceptional = []

    if etmc:
        if read2 != 'direct' and read3 != 'direct':
            metadata_file = Path('../%s/corr/%s/metadata_%s.pckl' % (
                params['datadir']['mydata'], params['ensemble'], params['ensemble']))
            with metadata_file.open('rb') as metadata_stream:
                params['confs'] = pickle.load(metadata_stream)['confs']
        else:
            root = Path('../data/traceless_operators')
            ensembles = sorted(root.glob('%s*' % params['ensemble']))
            file_labels = {
                'pion': 'pion_uins',
                'kaon': 'kaon_uins',
                'kaon_s': 'kaon_sins',
            }

            # The same number in a and b denotes two different configurations.
            params['confs'] = []
            for ensemble_dir in ensembles:
                confs = {}
                for hadron, file_label in file_labels.items():
                    files = ensemble_dir.glob('*_%s.h5' % file_label)
                    confs[hadron] = {
                        int(file.name.split('_', 1)[0]) for file in files
                    }

                common_confs = confs['pion'] & confs['kaon'] & confs['kaon_s']
                for conf in sorted(common_confs):
                    identity = (ensemble_dir.name, conf)
                    if identity in exceptional:
                        continue

                    valid = True
                    for file_label in file_labels.values():
                        file = ensemble_dir / ('%04d_%s.h5' % (conf, file_label))
                        with h5.File(file, 'r') as three_file:
                            file_dt_list = [int(value) for value in three_file.attrs['dt_list']]
                            if not all(dt in file_dt_list for dt in params['tsep_list']):
                                valid = False
                                break

                            source_names = set(three_file['O4/dt%d' % params['tsep_list'][0]].keys())
                            for dt in params['tsep_list'][1:]:
                                source_names &= set(three_file['O4/dt%d' % dt].keys())
                            if not source_names:
                                valid = False
                                break

                    if valid:
                        params['confs'].append(identity)

    params['N'] = len(params['confs'])
    params['hca'] = 197.3269804 / params['spacing']

    # Binning setting (Nb: N after binning)
    params['bin'] = True
    params['Nb'] = 60
    if not params['bin']: params['Nb'] = params['N']

    # Resampling setting
    if params['tech'] == 'jackknife':
        params['relen'] = params['Nb'] + 1
    if params['tech'] == 'bootstrap':
        params['boots_num'] = 2000
        #params['boots_num'] = 400
        params['Nbs'] = params['boots_num']
        params['Mbs'] = params['Nb']
        params['seed'] = 0
        params['relen'] = params['boots_num'] + 1

    return params

latcha = {'etacetac_A1_sp':r'$A_1$',
          'jpsijpsi_E_sp':r'$E$',
          'jpsijpsi_T2_sp':r'$T_2$'}

iso_dict = {'0': r'0',
            '1': r'1',
            '2': r'2',
            '1d2': r'1/2',
            '3d2': r'3/2',
            '30': r'0'}
irrep_dict = {'A1+': r'A_1^+',
            'T1-': r'T_1^-',
            'T1+': r'T_1^+',
            'T2+': r'T_2^+',
            'E+': r'E^+',
            'A1': r'A_1',
            'A2': r'A_2',
            'E2': r'E_2',
            'B1': r'B_1',
            'B2': r'B_2'}

mrkr = {0:'v', 1:'s', 2:'o', 3:'d', 4:'^', 5:'X', 6:'P'}
mrkr_offset = {1:0, 2:.1, 3:.2, 4:.3}
clrs = ['r','g','b','k','orange','DeepPink','blueviolet','RoyalBlue','MediumBlue',
        'LightPink','LavenderBlush','PaleVioletRed','HotPink','DeepPink','MediumVioletRed','Orchid','Thistle',
        'Violet','Magenta','Purple','MediumOrchid','DarkOrchid','Indigo','BlueViolet','MediumSlateBlue']
def clrscm(N,i):
    return cm.rainbow(np.linspace(0,1,N))[i]

alphas = {1:0.6, 2:.6, 3:.6, 4:.6, 5:.6, 6:.6, 7:.6}

def labels(k):
    if k == 'pion': return r'\pi'
    if k == 'kaon': return r'K'
    if k == 'kaon_s': return r'K_s'
    def label_translate(kpart):
        par_mom = kpart.split('_')
        one_label = {}
        two_label = {}

        one_label['D'] = r'D'
        one_label['Dst'] = r'D^{*}'
        one_label['Dst2'] = r'D^{*}2'
        one_label['D0st'] = r'D^{0*}'
        one_label['D0st2'] = r'D^{0*}2'
        one_label['D1'] = r'D_1'
        one_label['D12'] = r'D_12'
        one_label['1D'] = r'1D'

        one_label['Dst1D'] = r'D^{*}1D'
        one_label['D0st1D'] = r'D^{0*}1D'

        one_label['etac'] = r'\eta_c'
        one_label['jpsi'] = r'J/\psi'
        one_label['chic0'] = r'\chi_{c0}'
        one_label['chic1'] = r'\chi_{c1}'

        one_label['pion'] = r'\pi'
        one_label['rho'] = r'\rho'
        one_label['a0'] = r'a_0'
        one_label['a1'] = r'a_1'
        one_label['sigma'] = r'\sigma'

        one_label['Ds'] = r'D_s'
        one_label['Dsst'] = r'D_s^{*}'
        one_label['Ds0s'] = r'D_{s0}^{*}'
        one_label['Ds1'] = r'D_{s1}'

        one_label['eta'] = r'\eta'

        one_label['GEVPed'] = r'GEVP'
        one_label['GEVPedR'] = r'GEVPR'
        one_label['RGEVPed'] = r'RGEVP'
        one_label['Rdiag'] = r'Rdiag'

        two_label['Dpi'] = [r'D', r'\pi']
        two_label['Dstpi'] = [r'D^*', r'\pi']
        two_label['Dstpi2'] = [r'D^*2', r'\pi']

        two_label['pipi'] = [r'\pi', r'\pi']

        if len(par_mom) == 1:
            if par_mom[0] in one_label:
                return r'%s'%(one_label[par_mom[0]])
        elif len(par_mom) == 2:
            if par_mom[0] in one_label:
                return r'%s(%s)'%(one_label[par_mom[0]],par_mom[1])
        elif len(par_mom) == 3:
            if par_mom[0] in two_label:
                return r'%s(%s)%s(%s)'%(two_label[par_mom[0]][0],par_mom[1],two_label[par_mom[0]][1],par_mom[2])
        # No matching
        return par_mom

    kpart = k.split('-')

    return k
